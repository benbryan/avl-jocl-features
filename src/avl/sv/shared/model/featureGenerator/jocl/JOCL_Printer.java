/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package avl.sv.shared.model.featureGenerator.jocl;

import java.util.ArrayList;
import org.jocl.CL;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clEnqueueReadBuffer;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_event;
import org.jocl.cl_mem;

/**
 *
 * @author benbryan
 */
public class JOCL_Printer {
    public static void print_cl_mem_Float(cl_command_queue commandQueue, cl_mem mem, int length, ArrayList<cl_event> waitForEvents) {
        float data[] = new float[length];
        cl_event readFeatures = new cl_event();
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        clEnqueueReadBuffer(commandQueue, mem,
                CL_TRUE, 0, length * Sizeof.cl_float,
                Pointer.to(data), waitForEventsArray.length, waitForEventsArray, readFeatures);
        waitForEvents.add(readFeatures);
        waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clWaitForEvents(waitForEventsArray.length, waitForEventsArray);
        System.out.println("Start Print");
        for (int i = 0; i < data.length; i++) {
            System.out.println(String.valueOf(data[i]));
        }
        System.out.println("End Print");
    }

    public static void print_cl_mem_byte(cl_command_queue commandQueue, cl_mem mem, int length, ArrayList<cl_event> waitForEvents) {
        byte data[] = new byte[length];
        cl_event readFeatures = new cl_event();
        cl_event[] waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        clEnqueueReadBuffer(commandQueue, mem,
                CL_TRUE, 0, length,
                Pointer.to(data), waitForEventsArray.length, waitForEventsArray, readFeatures);
        waitForEvents.add(readFeatures);
        waitForEventsArray = waitForEvents.toArray(new cl_event[waitForEvents.size()]);
        CL.clWaitForEvents(waitForEventsArray.length, waitForEventsArray);
        System.out.println("Start Print");
        for (int i = 0; i < data.length; i++) {
            System.out.println(String.valueOf(data[i]));
        }
        System.out.println("End Print");
    }
}
