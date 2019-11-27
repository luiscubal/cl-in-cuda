package pt.up.fe.specs.clcuda;

import java.util.ArrayList;
import java.util.List;

public class ProgramStats {
	public String buildOpts;
	private StringBuilder buildLog = new StringBuilder();
	public List<KernelStats> kernels = new ArrayList<>();
	
	public String getBuildLog() {
		return buildLog.toString();
	}
	
	public String toToml() {
		StringBuilder builder = new StringBuilder();
		
		builder.append("[program]\n");
		builder.append("build_log = \"");
		builder.append(quote(getBuildLog()));
		builder.append("\"\n");
		builder.append("build_options = \"");
		builder.append(quote(buildOpts));
		builder.append("\"\n\n");
		
		for (KernelStats kernelStat : kernels) {
			builder.append("[[kernels]]\n");
			builder.append("name = \"");
			builder.append(quote(kernelStat.name));
			builder.append("\"\n");
			builder.append("symbol_name = \"");
			builder.append(quote(kernelStat.launcherSymbolName));
			builder.append("\"\n\n");
			
			for (ArgumentStats arg : kernelStat.args) {
				builder.append("[[kernels.args]]\n");
				switch (arg.type) {
				case GLOBAL_PTR:
					builder.append("type = \"global_ptr\"\n\n");
					break;
				case LOCAL_PTR:
					builder.append("type = \"local_ptr\"\n\n");
					break;
				case SCALAR:
					builder.append("type = \"scalar\"\n");
					builder.append("size = ");
					builder.append(arg.size);
					builder.append("\n\n");
				}
			}
		}
		
		return builder.toString();
	}
	
	private static String quote(String s) {
		return s.replace("\\", "\\\\")
				.replace("\r", "\\r")
				.replace("\n", "\\n");
	}
}
