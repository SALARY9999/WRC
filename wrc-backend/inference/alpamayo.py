"""The Alpamayo checkpoint, its prompt and the sampling call.

Everything the model call depends on lives here so that the call itself is
reproducible from one file: the checkpoint and dtype, the chat message built by
the official ``helper.create_message``, the processor, the sampling parameters,
and the fingerprint of the exact tokenized input (images included).

What this module deliberately does not do:

* it does not read the pipeline's state, the planned route, the takeover
  decision or the waypoint sender - ``infer`` takes the camera window and the
  ego tensors as arguments and returns a plain :class:`ModelResult`;
* it does not validate the trajectory, decide control, or crop anything.  A
  model output that is unusable is returned as-is; the caller rejects it, so
  the reject reason stays in one place.

Calling convention is a compatibility surface
    ``torch.bfloat16`` + ``attn_implementation="sdpa"`` + ``.to("cuda")``, the
    HF mirror environment defaults, ``top_p``/``temperature``/``num_traj_samples``
    /``max_generation_length`` all come from settings, and the prompt is the
    official one (no local takeover prompt).  ``tools/infer_images_pose_only.py``
    reaches into ``_processor`` and ``_model`` to replay a package on a GPU box;
    keep those attribute names.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from camera.models import CameraWindow
    from config.loader import AppConfig
    from vehicle.models import PoseSample

logger = logging.getLogger(__name__)

def _messages_text(messages: Any) -> str:
    """Text parts of the chat messages (images are dropped).

    This is the human-readable half of the input fingerprint: the package
    records it so that "what did the model actually read" can be answered
    without decoding token ids.
    """
    chunks: list[str] = []
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    return "\n".join(x for x in chunks if x)


@dataclass
class ModelResult:
    cot: str
    positions: np.ndarray  # (T, 3)
    rotations: np.ndarray  # (T, 3, 3)
    t0_us: int
    ref_pose: PoseSample
    # Fingerprint of the exact model input (token ids, images included) plus the
    # human-readable text part, so a package proves what the model really saw.
    input_ids_sha256: str = ""
    input_ids_shape: tuple[int, ...] = ()
    prompt_text: str = ""


class AlpamayoRunner:
    """One complete Alpamayo instance on one device.

    ``device`` is explicit rather than hardcoded to ``"cuda"``.  In the
    multi-GPU pool each worker process is pinned by ``CUDA_VISIBLE_DEVICES`` and
    passes ``"cuda:0"``; in the original single-GPU path the default keeps the
    historical behaviour, because there ``CUDA_VISIBLE_DEVICES`` is set by
    ``run_backend_real.sh`` and ``"cuda"`` already means "the one card I was
    given".  A runner never *chooses* a device - it is told which one.
    """

    def __init__(self, settings: AppConfig, device: str = "cuda") -> None:
        self.settings = settings
        self.device = str(device)
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    @property
    def device_type(self) -> str:
        """``"cuda"`` for both ``"cuda"`` and ``"cuda:0"``.

        ``torch.autocast`` takes a device *type*, not an index, and rejects
        ``"cuda:0"`` - so the two spellings have to be separated once, here,
        rather than at each call site.
        """
        return torch.device(self.device).type

    @property
    def loaded(self) -> bool:
        """Whether the checkpoint has been loaded.

        Read-only and cheap, so the pipeline can publish it every cycle.  It is
        False until the first ``infer()`` reaches ``ensure_loaded()``, which is
        what lets readiness distinguish "loop alive, model never loaded" - a
        state ``/health`` cannot express.  It does not read ``self._processor``:
        the two are set together under the lock, and ``_model`` is the one the
        caller's ``if self._model is not None`` short-circuit already trusts.
        """
        return self._model is not None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
            # The mirror comes from ``network.http_client.hf_endpoint``, which
            # ``HF_ENDPOINT`` in the environment overrides - the same precedence
            # the loader applies everywhere else, and the same value this line
            # used to hardcode.
            os.environ.setdefault(
                "HF_ENDPOINT", self.settings.network.hf_endpoint
            )
            from alpamayo1_5 import helper
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

            # On PGX the checkpoint and processor are transferred separately.
            # The official helper otherwise defaults to a remote processor ID,
            # which cannot be resolved with HF_HUB_OFFLINE=1. Keep the existing
            # behaviour for deployments that do not set this path.
            processor_path = os.environ.get("ALPAMAYO_PROCESSOR_PATH", "").strip()
            if processor_path:
                if not os.path.isdir(processor_path):
                    raise FileNotFoundError(
                        f"ALPAMAYO_PROCESSOR_PATH does not exist: {processor_path}"
                    )
                helper.BASE_PROCESSOR_NAME = processor_path
                # The checkpoint config is complete. Avoid constructing its
                # unrelated default config, which looks up a remote processor.
                Alpamayo1_5.config_class.has_no_defaults_at_init = True
                logger.info("using local Alpamayo processor %s", processor_path)

            logger.info(
                "loading Alpamayo checkpoint %s device=%s",
                self.settings.checkpoint,
                self.device,
            )
            model = Alpamayo1_5.from_pretrained(
                self.settings.checkpoint,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                **({"local_files_only": True} if processor_path else {}),
            ).to(self.device)
            model.eval()
            self._processor = helper.get_processor(model.tokenizer)
            self._model = model
            logger.info("Alpamayo loaded device=%s", self.device)

    def infer(
        self,
        window: CameraWindow,
        ego_xyz: torch.Tensor,
        ego_rot: torch.Tensor,
        ref_pose: PoseSample,
    ) -> ModelResult:
        from alpamayo1_5 import helper

        self.ensure_loaded()
        assert self._model is not None and self._processor is not None
        settings = self.settings
        indices = torch.tensor(window.camera_indices, dtype=torch.int64)
        messages = helper.create_message(
            window.frames.flatten(0, 1),
            indices,
            num_frames_per_camera=settings.frames_per_camera,
            nav_text=settings.navigation_text or None,
        )
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Fingerprint the exact model input (token ids include the images).
        input_ids = inputs["input_ids"]
        input_ids_sha256 = hashlib.sha256(
            input_ids.cpu().numpy().astype(np.int32).tobytes()
        ).hexdigest()
        input_ids_shape = tuple(int(x) for x in input_ids.shape)
        prompt_text = _messages_text(messages)
        data = helper.to_device(
            {
                "tokenized_data": inputs,
                "ego_history_xyz": ego_xyz,
                "ego_history_rot": ego_rot,
            },
            self.device,
        )
        with torch.inference_mode(), torch.autocast(
            self.device_type, dtype=torch.bfloat16
        ):
            pred_xyz, pred_rot, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                data=data,
                top_p=settings.top_p,
                temperature=settings.temperature,
                num_traj_samples=settings.num_traj_samples,
                max_generation_length=settings.max_generation_length,
                return_extra=True,
            )

        xyz = pred_xyz[0, 0].float().cpu().numpy()  # (K, T, 3)
        rot = pred_rot[0, 0].float().cpu().numpy()
        positions = xyz[0] if xyz.ndim == 3 else xyz  # (T, 3)
        rotations = rot[0] if rot.ndim == 4 else rot

        cot = ""
        if "cot" in extra:
            arr = extra["cot"]
            if isinstance(arr, np.ndarray) and arr.size:
                cot = str(arr.reshape(-1)[0])
            elif isinstance(arr, list) and arr:
                cot = str(arr[0])
        return ModelResult(
            cot=cot,
            positions=np.asarray(positions, dtype=np.float64),
            rotations=np.asarray(rotations, dtype=np.float64),
            t0_us=window.t0_us,
            ref_pose=ref_pose,
            input_ids_sha256=input_ids_sha256,
            input_ids_shape=input_ids_shape,
            prompt_text=prompt_text,
        )


__all__ = ["AlpamayoRunner", "ModelResult"]
