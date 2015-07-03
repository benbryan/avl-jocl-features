package avl.sv.shared.model.featureGenerator.jocl.plane;

import avl.sv.shared.model.featureGenerator.jocl.KernelHelper;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.HashSet;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

public class PlaneGenerator {
    
    final cl_context context;
    final cl_command_queue commandQueue;
    private cl_kernel   clRGB_DeInterlaceKernel, 
                        clRGB_ToGrayKernel,
                        clRGB_DeInterlaceToHSVKernel;
    
    public PlaneGenerator(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        clRGB_DeInterlaceKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("RGB_DeInterlace_kernal.cl"), "rgb_DeInterlace_kernal");
        clRGB_DeInterlaceToHSVKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("RGB_DeInterlaceToHSV_kernal.cl"), "rgb_DeInterlaceToHSV_kernal");
        clRGB_ToGrayKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("rgb_ToGray_kernal.cl"), "rgb_ToGray_kernal");
    }
    
    @Override
    protected void finalize() throws Throwable {
        CL.clReleaseKernel(clRGB_DeInterlaceKernel);
        CL.clReleaseKernel(clRGB_DeInterlaceToHSVKernel);
        CL.clReleaseKernel(clRGB_ToGrayKernel);
        super.finalize(); //To change body of generated methods, choose Tools | Templates.
    }
    
    public ArrayList<CL_PlaneSet> getRequiredCL_PlaneSets(BufferedImage[] imgs, ArrayList<Plane> planes, HashSet<cl_event> waitForEvents){
        final CL_RGB_Set imgSetRGB = imagesToGPU(imgs, waitForEvents);
        ArrayList<CL_PlaneSet> cl_planeSetsTemp = new ArrayList<>();
        ArrayList<Plane.Names> planesToGenerate = new ArrayList<>();
        for (Plane plane:planes){
            planesToGenerate.add(plane.planeName);
        }
        while (!planesToGenerate.isEmpty()){
            if (planesToGenerate.contains(Plane.Names.Red) | planesToGenerate.contains(Plane.Names.Green) | planesToGenerate.contains(Plane.Names.Blue)){
                cl_planeSetsTemp.addAll(rgbDeInterlace(imgSetRGB, waitForEvents));
                planesToGenerate.remove(Plane.Names.Red);
                planesToGenerate.remove(Plane.Names.Green);
                planesToGenerate.remove(Plane.Names.Blue);
            }
            if (planesToGenerate.contains(Plane.Names.Hue) | planesToGenerate.contains(Plane.Names.Saturation) | planesToGenerate.contains(Plane.Names.Value)){
                cl_planeSetsTemp.addAll(rgbDeInterlaceToHSV(imgSetRGB, waitForEvents));
                planesToGenerate.remove(Plane.Names.Hue);
                planesToGenerate.remove(Plane.Names.Saturation);
                planesToGenerate.remove(Plane.Names.Value);
            }
            if (planesToGenerate.contains(Plane.Names.Gray)){
                cl_planeSetsTemp.addAll(rgbToGray(imgSetRGB, waitForEvents));
                planesToGenerate.remove(Plane.Names.Gray);
            }
        }
        
        ArrayList<CL_PlaneSet> cl_planeSets = new ArrayList<>();
        for (Plane plane:planes){
            for (int i = 0; i < cl_planeSetsTemp.size(); i++) {
                if (plane.planeName == cl_planeSetsTemp.get(i).planeName){
                    CL_PlaneSet temp = cl_planeSetsTemp.get(i);
                    cl_planeSets.add(temp);
                    break;
                }
            }
        }
        cl_planeSetsTemp.removeAll(cl_planeSets);
        for (CL_PlaneSet planeSet:cl_planeSetsTemp){
            planeSet.free();
        }
        imgSetRGB.free();
        return cl_planeSets;
    }

    private CL_RGB_Set imagesToGPU(final BufferedImage imgs[], HashSet<cl_event> waitForEvents) {
        final int w = imgs[0].getWidth();
        final int h = imgs[0].getHeight();
        final int bands = 3;
        final byte b[] = new byte[w * h * imgs.length * bands];
        for (int i = 0; i < imgs.length; i++) {
            final byte data[] = ((DataBufferByte) imgs[i].getRaster().getDataBuffer()).getData();
            System.arraycopy(data, 0, b, i * w * h * bands, w * h * bands);
        }
        cl_mem imgMem = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, b.length, null, null);
        cl_event event = new cl_event();
        CL.clEnqueueWriteBuffer(commandQueue, imgMem, true, 0, b.length, Pointer.to(b), 0, null, event);
        waitForEvents.add(event);
        
        CL_RGB_Set imgSet = new CL_RGB_Set(imgs.length, w, h, imgMem);
        return imgSet;
    }
    
    private ArrayList<CL_PlaneSet> rgbDeInterlace(CL_RGB_Set imgs, HashSet<cl_event> waitForEvents) {
        cl_kernel kernel = clRGB_DeInterlaceKernel;
        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, 0, 0};
        cl_mem outR = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * numelPixels, null, null);
        cl_mem outG = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * numelPixels, null, null);
        cl_mem outB = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * numelPixels, null, null);
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outR));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(outG));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(outB));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int2, Pointer.to(params));

        cl_event event = new cl_event();
        long localWorkSize[] = {imgs.w};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        waitForEvents.add(event);

        ArrayList<CL_PlaneSet> rgbPlanes = new ArrayList<>();
        rgbPlanes.add(new CL_PlaneSet(Plane.Names.Red, imgs.w, imgs.h, imgs.numOfImages, outR));
        rgbPlanes.add(new CL_PlaneSet(Plane.Names.Green, imgs.w, imgs.h, imgs.numOfImages, outG));
        rgbPlanes.add(new CL_PlaneSet(Plane.Names.Blue, imgs.w, imgs.h, imgs.numOfImages, outB));
        return rgbPlanes;
    }
    
    private ArrayList<CL_PlaneSet> rgbDeInterlaceToHSV(CL_RGB_Set imgs, HashSet<cl_event> waitForEvents) {
        cl_kernel kernel = clRGB_DeInterlaceToHSVKernel;
        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, 0, 0};
        cl_mem outH = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * numelPixels, null, null);
        cl_mem outS = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * numelPixels, null, null);
        cl_mem outV = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * numelPixels, null, null);
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outH));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(outS));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(outV));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int2, Pointer.to(params));

        cl_event event = new cl_event();
        long localWorkSize[] = {imgs.w};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        waitForEvents.add(event);

        ArrayList<CL_PlaneSet> rgbPlanes = new ArrayList<>();
        rgbPlanes.add(new CL_PlaneSet(Plane.Names.Hue, imgs.w, imgs.h, imgs.numOfImages, outH));
        rgbPlanes.add(new CL_PlaneSet(Plane.Names.Saturation, imgs.w, imgs.h, imgs.numOfImages, outS));
        rgbPlanes.add(new CL_PlaneSet(Plane.Names.Value, imgs.w, imgs.h, imgs.numOfImages, outV));
        return rgbPlanes;
    }   
    
    private ArrayList<CL_PlaneSet> rgbToGray(CL_RGB_Set imgs, HashSet<cl_event> waitForEvents) {
        cl_kernel kernel = clRGB_ToGrayKernel;
        int numelPixels = imgs.w * imgs.h;
        int params[] = {imgs.numOfImages, numelPixels, 0, 0};
        cl_mem grays = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * numelPixels, null, null);
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(imgs.imgMem));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(grays));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int2, Pointer.to(params));

        cl_event event = new cl_event();
        long localWorkSize[] = {imgs.w};
        long globalWorkSize[] = {localWorkSize[0] * imgs.numOfImages};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        waitForEvents.add(event);

        ArrayList<CL_PlaneSet> rgbPlanes = new ArrayList<>();
        rgbPlanes.add(new CL_PlaneSet(Plane.Names.Gray, imgs.w, imgs.h, imgs.numOfImages, grays));
        return rgbPlanes;
    }
    
}
