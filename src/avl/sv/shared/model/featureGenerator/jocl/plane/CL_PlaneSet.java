package avl.sv.shared.model.featureGenerator.jocl.plane;

import avl.sv.shared.model.featureGenerator.jocl.plane.Plane;
import org.jocl.CL;
import org.jocl.cl_mem;

public class CL_PlaneSet {

    public final Plane.Names planeName;
    public final int w, h, numOfImages;
    public final cl_mem imgMem;

    public CL_PlaneSet(Plane.Names name, int w, int h, int numOfImages, cl_mem imgMem) {
        this.planeName = name;
        this.w = w;
        this.h = h;
        this.numOfImages = numOfImages;
        this.imgMem = imgMem;
    }

    public void free() {
        CL.clReleaseMemObject(imgMem);
    }
}
