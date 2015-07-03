package avl.sv.shared.model.featureGenerator.jocl.feature;

import avl.sv.shared.model.featureGenerator.jocl.plane.CL_PlaneSet;
import avl.sv.shared.model.featureGenerator.jocl.KernelHelper;
import java.util.ArrayList;
import java.util.HashSet;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_event;
import org.jocl.cl_kernel;

public class Cooccurance {
    
    final cl_context context;
    final cl_command_queue commandQueue;
    private cl_kernel clContrastKernel, clCorrelationKernel, clHomogeneityKernel;

    public Cooccurance(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        clContrastKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Contrast_kernal.cl"), "contrast_kernal");
        clCorrelationKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Correlation_kernal.cl"), "correlation_kernal");
        clHomogeneityKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Homogeneity_kernal.cl"), "homogeneity_kernal");
    }
    
    @Override
    protected void finalize() throws Throwable {
        CL.clReleaseKernel(clContrastKernel);
        CL.clReleaseKernel(clCorrelationKernel);
        CL.clReleaseKernel(clHomogeneityKernel);
        super.finalize(); //To change body of generated methods, choose Tools | Templates.
    }

    public ArrayList<CL_Feature> contrast(CL_PlaneSet imgs, HashSet<cl_event> waitForEvents) {
        cl_kernel kernel = clContrastKernel;
        CL_Feature outR = new CL_Feature();
        CL_Feature outD = new CL_Feature();

        outR.name = Feature.Names.ContrastR;
        outD.name = Feature.Names.ContrastD;

        outR.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);
        outD.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, imgs.w, 0};
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outR.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(outD.cl_data));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_float * imgs.w, null);
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float * imgs.w, null);
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_int * params.length, Pointer.to(params));

        cl_event event = new cl_event();

        long localWorkSize[] = {imgs.w};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        outR.cl_event = event;
        outD.cl_event = event;

//        waitForEvents.add(event);
//        print_cl_mem_Float(outR.cl_data, imgs.numOfImages, waitForEvents);
//        print_cl_mem_byte(imgs.imgMem, 10, waitForEvents);
        ArrayList<CL_Feature> features = new ArrayList<>();
        features.add(outR);
        features.add(outD);
        return features;
    }

    public ArrayList<CL_Feature> homogeneity(CL_PlaneSet imgs, HashSet<cl_event> waitForEvents) {
        cl_kernel kernel = clHomogeneityKernel;
        CL_Feature outR = new CL_Feature();
        CL_Feature outD = new CL_Feature();

        outR.name = Feature.Names.HomogeneityR;
        outD.name = Feature.Names.HomogeneityD;

        outR.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);
        outD.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, imgs.w, 0};
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outR.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(outD.cl_data));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_float * imgs.w, null);
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float * imgs.w, null);
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_int * params.length, Pointer.to(params));

        cl_event event = new cl_event();

        long localWorkSize[] = {imgs.w};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        outR.cl_event = event;
        outD.cl_event = event;

//        waitForEvents.add(event);
//        print_cl_mem_Float(outR.cl_data, imgs.numOfImages, waitForEvents);
//        print_cl_mem_byte(imgs.imgMem, 10, waitForEvents);
        ArrayList<CL_Feature> features = new ArrayList<>();
        features.add(outR);
        features.add(outD);
        return features;
    }

    public ArrayList<CL_Feature> correlation(CL_PlaneSet imgs, HashSet<cl_event> waitForEvents, CL_Feature mean, CL_Feature moment2) {
        cl_kernel kernel = clCorrelationKernel;
        CL_Feature outR = new CL_Feature();
        CL_Feature outD = new CL_Feature();

        outR.name = Feature.Names.CorrelationR;
        outD.name = Feature.Names.CorrelationD;

        outR.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);
        outD.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, imgs.w, 0};
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(mean.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(moment2.cl_data));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(outR.cl_data));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(outD.cl_data));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_float * imgs.w, null);
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_float * imgs.w, null);
        CL.clSetKernelArg(kernel, 7, Sizeof.cl_int * params.length, Pointer.to(params));

        cl_event event = new cl_event();

        long localWorkSize[] = {imgs.w};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        outR.cl_event = event;
        outD.cl_event = event;

        ArrayList<CL_Feature> features = new ArrayList<>();
        features.add(outR);
        features.add(outD);
        return features;
    }    
    
}
