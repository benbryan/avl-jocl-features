package avl.sv.shared.model.featureGenerator.jocl;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_device_id;

public class Device {
    public final cl_device_id deviceID;
    public final String deviceName;

    public Device(cl_device_id device_Id) {
        this.deviceID = device_Id;
        deviceName = getDeviceName(device_Id);
    }

    private String getDeviceName(cl_device_id deviceID){
        int param_value_size = 100;
        byte param_value[] = new byte[param_value_size];
        long param_value_size_ret[] = new long[1];
        CL.clGetDeviceInfo(deviceID, CL.CL_DEVICE_NAME, param_value_size, Pointer.to(param_value), param_value_size_ret);
        return new String(param_value).trim();
    }

    @Override
    public String toString() {
        return deviceName;
    }

}
