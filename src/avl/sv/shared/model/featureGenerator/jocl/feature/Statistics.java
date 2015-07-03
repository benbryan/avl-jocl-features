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

public class Statistics {
    
    final cl_context context;
    final cl_command_queue commandQueue;
    private cl_kernel   clMeansKernel, clMomentsKernel, clKurtosisSkewnessKernal;
   
    public Statistics(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        clMeansKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Means_kernal.cl"), "means_kernal");
        clMomentsKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Moments_kernal.cl"), "moments_kernal");
        clKurtosisSkewnessKernal = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("KurtosisSkewness_kernal.cl"), "kurtosisSkewness_kernal");
    }
    
    @Override
    protected void finalize() throws Throwable {
        CL.clReleaseKernel(clMeansKernel);
        CL.clReleaseKernel(clMomentsKernel);
        CL.clReleaseKernel(clKurtosisSkewnessKernal);
        super.finalize(); //To change body of generated methods, choose Tools | Templates.
    }

    public CL_Feature mean(final CL_PlaneSet imgs, HashSet<cl_event> waitForEvents) {
        cl_kernel kernel = clMeansKernel;
        CL_Feature feature = new CL_Feature();
        feature.name = Feature.Names.Mean;

        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels};
        feature.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(feature.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int2, Pointer.to(params));

        long localWorkSize[] = {1};
        long globalWorkSize[] = {imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, feature.cl_event);
        waitForEvents.add(feature.cl_event);
//        JOCL_Printer.print_cl_mem_Float(commandQueue, feature.cl_data, imgs.numOfImages, waitForEvents);
//        JOCL_Printer.print_cl_mem_byte(commandQueue, imgs.imgMem, imgs.h*imgs.w*imgs.numOfImages, waitForEvents);
        return feature;
    }
    public ArrayList<CL_Feature> moments(CL_PlaneSet imgs, HashSet<cl_event> waitForEvents, CL_Feature means) {
        cl_kernel kernel = clMomentsKernel;
        CL_Feature moment2 = new CL_Feature();
        CL_Feature moment3 = new CL_Feature();

        moment2.name = Feature.Names.Moment2;
        moment3.name = Feature.Names.Moment3;

        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, imgs.w, 0};

        moment2.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);
        moment3.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(means.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(moment2.cl_data));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(moment3.cl_data));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int4, Pointer.to(params));

        cl_event event = new cl_event();
        long localWorkSize[] = {imgs.w, 1};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages, 1};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        moment2.cl_event = event;
        moment3.cl_event = event;

        ArrayList<CL_Feature> features = new ArrayList<>();
        features.add(moment2);
        features.add(moment3);

        return features;
    }
    public ArrayList<CL_Feature> kurtosisSkewness(CL_PlaneSet imgs, HashSet<cl_event> waitForEvents, CL_Feature means) {
        cl_kernel kernel = clKurtosisSkewnessKernal;
        CL_Feature kurtosis = new CL_Feature();
        CL_Feature skewness = new CL_Feature();

        kurtosis.name = Feature.Names.Kurtosis;
        skewness.name = Feature.Names.Skewness;

        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, imgs.w, 0};

        kurtosis.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);
        skewness.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(means.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(kurtosis.cl_data));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(skewness.cl_data));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int4, Pointer.to(params));

        cl_event event = new cl_event();
        long localWorkSize[] = {imgs.w, 1};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages, 1};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        kurtosis.cl_event = event;
        skewness.cl_event = event;

        ArrayList<CL_Feature> features = new ArrayList<>();
        features.add(kurtosis);
        features.add(skewness);

        return features;
    }
    
}
