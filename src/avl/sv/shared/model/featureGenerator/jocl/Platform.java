package avl.sv.shared.model.featureGenerator.jocl;

import java.util.ArrayList;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

public class Platform {

    public final cl_platform_id platformID;
    public final String platformName;
    private final long deviceType = CL.CL_DEVICE_TYPE_ALL;
    public final ArrayList<Device> devices;

    public Platform(cl_platform_id platformID) {
        this.platformID = platformID;
        platformName = getPlatformName(platformID);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        CL.clGetDeviceIDs(platformID, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        devices = new ArrayList<>(numDevices);
        cl_device_id device_Ids[] = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platformID, deviceType, numDevices, device_Ids, null);
        for (int i = 0; i < numDevices; i++) {
            cl_device_id deviceID = device_Ids[i];
            devices.add(new Device(deviceID));
        }

    }

    private String getPlatformName(cl_platform_id platformID) {
        int param_value_size = 100;
        byte param_value[] = new byte[param_value_size];
        long param_value_size_ret[] = new long[1];
        CL.clGetPlatformInfo(platformID, CL.CL_PLATFORM_NAME, param_value_size, Pointer.to(param_value), param_value_size_ret);
        return new String(param_value).trim();
    }

    @Override
    public String toString() {
        return platformName;
    }
}
