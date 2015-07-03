package avl.sv.shared.model.featureGenerator.jocl;

import avl.sv.shared.model.featureGenerator.jocl.feature.Feature;
import avl.sv.shared.model.featureGenerator.jocl.plane.CL_PlaneSet;
import avl.sv.shared.model.featureGenerator.jocl.feature.GaborKernal;
import avl.sv.shared.model.featureGenerator.jocl.feature.Gabor;
import avl.sv.shared.model.featureGenerator.jocl.plane.Plane;
import avl.sv.shared.model.featureGenerator.jocl.plane.PlaneGenerator;
import avl.sv.shared.model.featureGenerator.jocl.feature.CL_Feature;
import avl.sv.shared.model.featureGenerator.jocl.feature.Cooccurance;
import avl.sv.shared.model.featureGenerator.jocl.feature.Statistics;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jocl.*;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_platform_id;

public class FeatureGeneratorJOCL {

    private final int platformIndex, deviceIndex;

    private cl_command_queue commandQueue;
    private cl_context context = null;

    Gabor gabor;
    Statistics statistics;
    Cooccurance cooccurance;
    PlaneGenerator planeGenerator;

    private cl_kernel clMagnitudeKernel, clSumKernel;

    ArrayList<Plane> planes = new ArrayList<>();
    private long deviceMemoryTotal = -1;

    public FeatureGeneratorJOCL(final int platformIndex, final int deviceIndex) {
        this.platformIndex = platformIndex;
        this.deviceIndex = deviceIndex;

        // Setup Default features
        for (Plane.Names p : new Plane.Names[]{Plane.Names.Red, Plane.Names.Green, Plane.Names.Blue}) {
            ArrayList<Feature> features = new ArrayList<>();
            features.add(new Feature(Feature.Names.Mean));
            planes.add(new Plane(p, features));
        }
    }

    private void setupJOCL() {

        // The platform, device type and device number
        // that will be used
        final long deviceType = CL.CL_DEVICE_TYPE_ALL;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID 
        cl_device_id devices[] = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);

        // Create a command-queue for the selected device
        commandQueue = CL.clCreateCommandQueue(context, device, 0, null);

        planeGenerator = new PlaneGenerator(context, commandQueue);
        gabor = new Gabor(context, commandQueue);
        statistics = new Statistics(context, commandQueue);
        cooccurance = new Cooccurance(context, commandQueue);
        clMagnitudeKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Magnitude_kernal.cl"), "magnitude_kernal");
        clSumKernel = KernelHelper.loadAndCompileKernel(context, getClass().getResourceAsStream("Sum_kernal.cl"), "sum_kernal");
    }

    ExecutorService executor = Executors.newSingleThreadExecutor();

    private class CL_FeatureSet {

        final ArrayList<CL_Feature> cl_features = new ArrayList<>();
        final CL_PlaneSet cL_Plane;
        private final HashSet<cl_event> waitForEvents;

        public CL_FeatureSet(CL_PlaneSet cL_Plane, HashSet<cl_event> waitForEvents) {
            this.cL_Plane = cL_Plane;
            HashSet<cl_event> cloneEvents = new HashSet<>();
            cloneEvents.addAll(waitForEvents);
            this.waitForEvents = cloneEvents;
        }

        public ArrayList<CL_Feature> getFeatures(ArrayList<Feature.Names> featureNames) {
            ArrayList<CL_Feature> outFeatures = new ArrayList<>();
            for (Feature.Names featureName : featureNames) {
                CL_Feature feature = getFeature(featureName, waitForEvents);
                outFeatures.add(feature);
            }
            return outFeatures;
        }

        public void free() {
            for (CL_Feature cl_feature : cl_features) {
                CL.clReleaseMemObject(cl_feature.cl_data);
            }
        }

        public CL_Feature getFeature(Feature.Names featureName, HashSet<cl_event> waitForEvents) {
            for (CL_Feature feature : cl_features) {
                if (feature.name.equals(featureName)) {
                    return feature;
                }
            }
            ArrayList<CL_Feature> tempFeatures;
            CL_Feature tempFeature0, tempFeature1, tempFeature2;
            switch (featureName) {
                case Mean:
                    tempFeature0 = statistics.mean(cL_Plane, waitForEvents);
                    cl_features.add(tempFeature0);
                    break;
                case Moment2: // Fall through
                case Moment3:
                    tempFeature0 = getFeature(Feature.Names.Mean, waitForEvents);
                    tempFeatures = statistics.moments(cL_Plane, waitForEvents, tempFeature0);
                    cl_features.addAll(tempFeatures);
                    break;
                case Kurtosis: // Fall through
                case Skewness:
                    tempFeature0 = getFeature(Feature.Names.Mean, waitForEvents);
                    tempFeatures = statistics.kurtosisSkewness(cL_Plane, waitForEvents, tempFeature0);
                    cl_features.addAll(tempFeatures);
                    break;
                case ContrastD: // Fall through
                case ContrastR:
                    tempFeatures = cooccurance.contrast(cL_Plane, waitForEvents);
                    cl_features.addAll(tempFeatures);
                    break;
                case HomogeneityD: // Fall through
                case HomogeneityR:
                    tempFeatures = cooccurance.homogeneity(cL_Plane, waitForEvents);
                    cl_features.addAll(tempFeatures);
                    break;
                case CorrelationD: // Fall through
                case CorrelationR:
                    tempFeature0 = getFeature(Feature.Names.Mean, waitForEvents);
                    tempFeature1 = getFeature(Feature.Names.Moment2, waitForEvents);
                    tempFeatures = cooccurance.correlation(cL_Plane, waitForEvents, tempFeature0, tempFeature1);
                    cl_features.addAll(tempFeatures);
                    break;
                case CorrelationMagnitude:
                    tempFeature0 = getFeature(Feature.Names.CorrelationD, waitForEvents);
                    tempFeature1 = getFeature(Feature.Names.CorrelationR, waitForEvents);
                    tempFeature2 = magnitude(cL_Plane, waitForEvents, tempFeature0, tempFeature1);
                    tempFeature2.name = Feature.Names.CorrelationMagnitude;
                    cl_features.add(tempFeature2);
                    break;
                case HomogeneityMagnitude:
                    tempFeature0 = getFeature(Feature.Names.HomogeneityD, waitForEvents);
                    tempFeature1 = getFeature(Feature.Names.HomogeneityR, waitForEvents);
                    tempFeature2 = magnitude(cL_Plane, waitForEvents, tempFeature0, tempFeature1);
                    tempFeature2.name = Feature.Names.HomogeneityMagnitude;
                    cl_features.add(tempFeature2);
                    break;
                case ContrastMagnitude:
                    tempFeature0 = getFeature(Feature.Names.ContrastD, waitForEvents);
                    tempFeature1 = getFeature(Feature.Names.ContrastR, waitForEvents);
                    tempFeature2 = magnitude(cL_Plane, waitForEvents, tempFeature0, tempFeature1);
                    tempFeature2.name = Feature.Names.ContrastMagnitude;
                    cl_features.add(tempFeature2);
                    break;
                case CorrelationSum:
                    tempFeature0 = getFeature(Feature.Names.CorrelationD, waitForEvents);
                    tempFeature1 = getFeature(Feature.Names.CorrelationR, waitForEvents);
                    tempFeature2 = fSum(cL_Plane, waitForEvents, tempFeature0, tempFeature1);
                    tempFeature2.name = Feature.Names.CorrelationSum;
                    cl_features.add(tempFeature2);
                    break;
                case HomogeneitySum:
                    tempFeature0 = getFeature(Feature.Names.HomogeneityD, waitForEvents);
                    tempFeature1 = getFeature(Feature.Names.HomogeneityR, waitForEvents);
                    tempFeature2 = fSum(cL_Plane, waitForEvents, tempFeature0, tempFeature1);
                    tempFeature2.name = Feature.Names.HomogeneitySum;
                    cl_features.add(tempFeature2);
                    break;
                case ContrastSum:
                    tempFeature0 = getFeature(Feature.Names.ContrastD, waitForEvents);
                    tempFeature1 = getFeature(Feature.Names.ContrastR, waitForEvents);
                    tempFeature2 = fSum(cL_Plane, waitForEvents, tempFeature0, tempFeature1);
                    tempFeature2.name = Feature.Names.ContrastSum;
                    cl_features.add(tempFeature2);
                    break;
                case Gabor_d0r0:
                case Gabor_d0r1:
                case Gabor_d0r2:
                case Gabor_d1r0:
                case Gabor_d1r1:
                case Gabor_d1r2:
                case Gabor_d2r0:
                case Gabor_d2r1:
                case Gabor_d2r2:
                case Gabor_d3r0:
                case Gabor_d3r1:
                case Gabor_d3r2:
                    tempFeature0 = gabor.getFeature(cL_Plane, gaborKernal(featureName), waitForEvents);
                    tempFeature0.name = featureName;
                    cl_features.add(tempFeature0);
                    break;
                default:
                    throw new IllegalArgumentException(featureName.name() + " is Not a valid feature name");
            }
            for (CL_Feature feature : cl_features) {
                if (feature.name.equals(featureName)) {
                    return feature;
                }
            }
            throw new Error("Failed to generate a feature");
        }
    }

    public static GaborKernal gaborKernal(Feature.Names featureName) {
        String name = featureName.name();
        name = name.split("_")[1];
        String temp[] = name.split("r");
        double dir = Double.parseDouble(temp[0].replace("d", ""));
        double r = Double.parseDouble(temp[1]);

        int dim = 21;
        double waveLength = (r + 1) / 5 * dim;
        double bandwidth = dim / 10;
        int kernelSize = (int) (bandwidth * 3 * (1 + r)); // must be odd number
        if (kernelSize % 2 != 1) {
            kernelSize++;
        }
        double aspectRatio = 1;
        double phaseOffset = 0;
        double orientation = dir * Math.PI / 4;

        return new GaborKernal(kernelSize, waveLength, bandwidth, aspectRatio, phaseOffset, orientation);
    }

    private class CallableImpl implements Callable<double[][]> {

        public Throwable ex = null;
        final private BufferedImage imgs[];

        public CallableImpl(BufferedImage[] imgs) {
            this.imgs = imgs;
        }
        
        @Override
        public double[][] call() throws Exception {
            try {
                if (context == null) {
                    setupJOCL();
                }
                CL.setExceptionsEnabled(true);
                double featuresData[][] = new double[imgs.length][getNumberOfFeatures()];
                HashSet<cl_event> waitForEvents = new HashSet<>();
                ArrayList<CL_PlaneSet> cl_planeSets = planeGenerator.getRequiredCL_PlaneSets(imgs, planes, waitForEvents);
                int featureIdx = 0;
                for (final CL_PlaneSet cl_planeSet : cl_planeSets) {
                    // print_cl_mem_byte(plane.imgMem, plane.w*plane.h, waitForEvents);
                    ArrayList<Feature.Names> featureNames = new ArrayList<>();
                    for (Plane plane : planes) {
                        if (plane.planeName.equals(cl_planeSet.planeName)) {
                            ArrayList<Feature> temp = plane.features;
                            for (Feature t : temp) {
                                featureNames.add(t.featureName);
                            }
                        }
                    }
                    ArrayList<Feature> features = generateFeatures(cl_planeSet, featureNames, waitForEvents);
                    for (int f = 0; f < featureNames.size(); f++) {
                        Feature.Names featureName = featureNames.get(f);
                        for (Feature feature : features) {
                            if (feature.featureName.equals(featureName)) {
                                for (int s = 0; s < imgs.length; s++) {
                                    featuresData[s][featureIdx] = feature.data[s];
                                }
                                featureIdx++;
                            }
                        }
                    }
                    cl_planeSet.free();
                }
                for (cl_event event : waitForEvents) {
                    if (!"cl_event[0x0]".equals(event.toString())) {
                        CL.clReleaseEvent(event);
                    }
                }
                return featuresData;
            } catch (Exception | Error ex) {
                this.ex = ex;
                return null;
            }
        }

    }
    
    public double[][] getFeaturesForImages(final BufferedImage[] imgs) throws Throwable {
        CallableImpl callable = new CallableImpl(imgs);
        Future<double[][]> future = executor.submit(callable);
        double[][] result = future.get(2, TimeUnit.MINUTES);
        if (callable.ex != null) {
            throw callable.ex;
        }
        return result;
    }

    /**
     * Names of the features set to be generated in the format
     *
     * @return
     * <Plane Name>-><Feature Name>
     * Plane names are from the enum Plane.Name Feature Names are from the enum
     * Feature.Name
     */
    public String[] getFeatureNames() {
        ArrayList<String> names = new ArrayList<>();
        for (Plane plane : planes) {
            for (Feature feature : plane.features) {
                names.add(plane.planeName + "->" + feature.featureName);
            }
        }
        return names.toArray(new String[names.size()]);
    }

    /**
     * Sets the features to be generated from a list of strings in the format
     * <Plane Name>-><Feature Name>
     *
     * @param featureNames Plane names are from the enum Plane.Name Feature
     * Names are from the enum Feature.Name
     */
    public void setFeatureNames(String[] featureNames) {
        planes = new ArrayList<>();
        for (String featureString : featureNames) {

            String[] temp = featureString.split("->");
            if (temp.length != 2) {
                continue;
            }
            Plane.Names planeName = Plane.Names.valueOf(temp[0]);
            Feature.Names featureName = Feature.Names.valueOf(temp[1]);
            if ((planeName != null) && (featureName != null)) {
                boolean featureAdded = false;
                addFeature:
                for (Plane plane : planes) {
                    if (plane.planeName.equals(planeName)) {
                        for (Feature feature : plane.features) {
                            if (feature.featureName.equals(featureName)) {
                                featureAdded = true;
                                break addFeature;
                            }
                        }
                        plane.features.add(new Feature(featureName));
                        featureAdded = true;
                        break addFeature;
                    }
                }
                if (featureAdded) {
                    continue;
                }
                ArrayList<Feature> features = new ArrayList<>();
                features.add(new Feature(featureName));
                planes.add(new Plane(planeName, features));
            }
        }
    }

    /**
     * Sets the planes with features to be generated directly
     *
     * @param planes
     */
    public void setPlanes(ArrayList<Plane> planes) {
        this.planes = planes;
    }

    public int getNumberOfFeatures() {
        int count = 0;
        for (Plane plane : planes) {
            count += plane.features.size();
        }
        return count;
    }

    private static long round(long groupSize, long globalSize) {
        long r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        CL.clReleaseKernel(clMagnitudeKernel);
        CL.clReleaseKernel(clSumKernel);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);
        super.finalize();
    }

    public CL_Feature magnitude(CL_PlaneSet imgs, HashSet<cl_event> waitForEvents, CL_Feature f1, CL_Feature f2) {
        cl_kernel kernel = clMagnitudeKernel;
        CL_Feature fOut = new CL_Feature();
        fOut.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        int params[] = {imgs.numOfImages, 0};
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(f1.cl_data));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(f2.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(fOut.cl_data));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int * params.length, Pointer.to(params));

        cl_event event = new cl_event();

        long localWorkSize[] = {16};
        long globalWorkSize[] = {round(localWorkSize[0], imgs.numOfImages)};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        fOut.cl_event = event;
        return fOut;
    }

    public CL_Feature fSum(CL_PlaneSet imgs, HashSet<cl_event> waitForEvents, CL_Feature f1, CL_Feature f2) {
        cl_kernel kernel = clSumKernel;
        CL_Feature fOut = new CL_Feature();
        fOut.cl_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, imgs.numOfImages * Sizeof.cl_float, null, null);

        int params[] = {imgs.numOfImages, 0};
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(f1.cl_data));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(f2.cl_data));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(fOut.cl_data));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int * params.length, Pointer.to(params));

        cl_event event = new cl_event();

        long localWorkSize[] = {16};
        long globalWorkSize[] = {round(localWorkSize[0], imgs.numOfImages)};
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, waitForEventsArray.length, waitForEventsArray, event);
        fOut.cl_event = event;

        return fOut;
    }

    private ArrayList<Feature> generateFeatures(CL_PlaneSet imgPlane, ArrayList<Feature.Names> featureNames, HashSet<cl_event> waitForEvents) {

        CL_FeatureSet featureSet = new CL_FeatureSet(imgPlane, waitForEvents);

        ArrayList<CL_Feature> CL_features = featureSet.getFeatures(featureNames);

        // Copy out features
        for (CL_Feature feature : CL_features) {
            waitForEvents.add(feature.cl_event);
        }
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        ArrayList<Feature> features = new ArrayList<>();
        for (CL_Feature cl_feature : CL_features) {
            float[] featureData = new float[imgPlane.numOfImages];
            cl_event readFeatures = new cl_event();
            CL.clEnqueueReadBuffer(commandQueue, cl_feature.cl_data,
                    CL.CL_TRUE, 0, imgPlane.numOfImages * Sizeof.cl_float,
                    Pointer.to(featureData), waitForEventsArray.length, waitForEventsArray, readFeatures);
            waitForEvents.add(readFeatures);
            Feature feature = new Feature(cl_feature.name);
            feature.data = featureData;
            features.add(feature);
        }
        HashSet<cl_event> eventsTemp = new HashSet<>();
        for (cl_event event : waitForEvents) {
            if (!"cl_event[0x0]".equals(event.toString())) {
                eventsTemp.add(event);
            }
        }
        waitForEvents = eventsTemp;
        waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        if (waitForEventsArray.length > 0) {
            CL.clWaitForEvents(waitForEventsArray.length, waitForEventsArray);
        }
        featureSet.free();
        return features;
    }

}
