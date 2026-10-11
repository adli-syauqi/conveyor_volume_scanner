import depthai as dai
from datetime import timedelta

device_infos = dai.Device.getAllConnectedDevices()
config = dai.HealthCheckConfig(
    powerSupplyCheckDuration=timedelta(seconds=20)
)

metrics = dai.Device.performHealthCheck(device_infos[0], config)
print(metrics)
print("")

with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    # Print DeviceID, USB speed, and available cameras on the device
    print('DeviceID:',device.getDeviceInfo().getDeviceId())
    print('USB speed:',device.getUsbSpeed())
    print('Connected cameras:',device.getConnectedCameras())

