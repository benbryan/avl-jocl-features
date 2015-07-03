package avl.sv.shared.model.featureGenerator.jocl.feature;

import avl.sv.shared.model.featureGenerator.jocl.plane.CL_PlaneSet;
import avl.sv.shared.model.featureGenerator.jocl.KernelHelper;
import java.util.HashSet;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

public class Gabor {
    
    final cl_context context;
    final cl_command_queue commandQueue;
    final cl_kernel kernel;

    public Gabor(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        kernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Gabor_kernal.cl"), "gabor_kernal");
    }
    
    @Override
    protected void finalize() throws Throwable {
        CL.clReleaseKernel(kernel);
        super.finalize(); //To change body of generated methods, choose Tools | Templates.
    }

    public CL_Feature getFeature(CL_PlaneSet imgs, GaborKernal gaborKernel, HashSet<cl_event> waitForEvents) {
        cl_event event = new cl_event();
        
        CL_Feature outGabor = new CL_Feature();

        outGabor.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        int numelKernelElements = gaborKernel.getKernalSize()*gaborKernel.getKernalSize();
        cl_mem cl_kernelReal = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, numelKernelElements * Sizeof.cl_float, null, null);
        cl_mem cl_kernelImag = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, numelKernelElements * Sizeof.cl_float, null, null);
        
        CL.clEnqueueWriteBuffer(commandQueue, cl_kernelReal, false, 0, numelKernelElements * Sizeof.cl_float, Pointer.to(gaborKernel.getReal()), 0, null, event);
        waitForEvents.add(event);
        CL.clEnqueueWriteBuffer(commandQueue, cl_kernelImag, false, 0, numelKernelElements * Sizeof.cl_float, Pointer.to(gaborKernel.getImag()), 0, null, event);
        waitForEvents.add(event);
        
        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, imgs.w, gaborKernel.getKernalSize()};
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(cl_kernelReal));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(cl_kernelImag));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(outGabor.cl_data));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float * imgs.w, null);
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_int * params.length, Pointer.to(params));

        long localWorkSize[] = {imgs.w};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        outGabor.cl_event = event;

//        waitForEvents.add(event);
//        print_cl_mem_Float(outR.cl_data, imgs.numOfImages, waitForEvents);
//        print_cl_mem_byte(imgs.imgMem, 10, waitForEvents);
        CL.clReleaseMemObject(cl_kernelImag);
        CL.clReleaseMemObject(cl_kernelReal);
        return outGabor;
    }
    
    
}
