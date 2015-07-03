
package avl.sv.shared.model.featureGenerator.jocl.plane;

import java.util.ArrayList;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_event;
import org.jocl.cl_mem;

public class CL_RGB_Set {

    final int w, h, numOfImages;
    final int bands = 3;
    final cl_mem imgMem;

    public CL_RGB_Set(int numOfImages, int width, int height, cl_mem imgMem) {
        w = width;
        h = height;
        this.numOfImages = numOfImages;
        this.imgMem = imgMem;
    }

    public void free() {
        CL.clReleaseMemObject(imgMem);
    }
}