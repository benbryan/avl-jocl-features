package avl.sv.shared.model.featureGenerator.jocl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jocl.CL;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

public class KernelHelper {
    private static String readKernel(InputStream kernalStream) {
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(kernalStream));
            StringBuilder sb = new StringBuilder();
            String line;
            while (true) {
                line = br.readLine();
                if (line == null) {
                    break;
                }
                sb.append(line).append("\n");
            }
            return sb.toString();
        } catch (IOException e) {
            Logger.getLogger(KernelHelper.class.getName()).log(Level.SEVERE, null, e);
            return null;
        }
    }
    public static cl_kernel loadAndCompileKernel(cl_context context, InputStream kernelStream, String kernelName) {
        // Create the OpenCL kernel from the program
        String source = readKernel(kernelStream);
        cl_program program = CL.clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        String compileOptions = "-cl-mad-enable";
        CL.clBuildProgram(program, 0, null, compileOptions, null, null);
        cl_kernel out = CL.clCreateKernel(program, kernelName, null);
        CL.clReleaseProgram(program);
        return out;
    }
}
