package avl.sv.shared.model.featureGenerator.jocl;

import org.jocl.cl_command_queue;
import org.jocl.cl_context;


abstract class Abstract_CL_Feature_Generator {
    protected final cl_context context;
    protected final cl_command_queue commandQueue;
    public Abstract_CL_Feature_Generator(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
    }
}
