"""Camera decoding and synchronization.

One input link.  The dependency graph is a line and not a cycle::

    camera.models       data only, no runtime imports
        ^
    camera.cimg         CIMG framing, stdlib only
        ^
    camera.cimg_source  the TCP+JPEG receiver
        ^
    camera.sync         buffers, synchronizer, window provider

``camera.models`` holds the types the receiver and the synchronizer share;
``camera.cimg`` is the JPEG-push protocol; ``camera.cimg_source`` pulls four
cameras out of one CIMG TCP stream; ``camera.sync`` turns that into the model's
window.

``tests/test_camera_import_structure.py`` asserts the edges that matter:

* ``camera.cimg`` and ``camera.cimg_source`` import neither ``camera.sync`` nor
  OpenCV at module scope.  Framing and queueing are pure stdlib, so the protocol
  can be tested against captured vehicle data with no decoder and no GPU, and a
  corrupt stream can be diagnosed without the heavy stack loaded.
"""
