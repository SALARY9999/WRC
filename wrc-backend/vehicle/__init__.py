"""Vehicle-side state: pose models, history, and the two inbound receivers.

``vehicle.models``
    The payload data classes (``GlobalPose``, ``ChassisInfo``, ``PoseSample``).
    Pure data; no runtime imports.
``vehicle.history``
    The ego history buffer and the tensor builders the model input uses.
``vehicle.receiver``
    The UDP side: ``UdpPoseListener``, ``MockPoseProvider`` and the payload
    decoders.
``vehicle.planned_route``
    The TCP side: the wire format (``<BBIIB`` header, ``<ff`` lon/lat points),
    the stream parser and the receiver.

``vehicle.planned_route`` imports the route data model (``PlannedRoute``,
``PlannedRoutePoint``, ``PlannedRouteCache``) from ``trajectory.takeover``
rather than defining it, so the dependency runs one way out of this package and
the wire decoder never sits underneath the decision engine.
"""
