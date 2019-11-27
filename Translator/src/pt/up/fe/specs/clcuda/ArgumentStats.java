package pt.up.fe.specs.clcuda;

public class ArgumentStats {
	public ArgumentType type;
	
	public static ArgumentStats fromGlobalPtr() {
		ArgumentStats stats = new ArgumentStats();
		stats.type = ArgumentType.GLOBAL_PTR;
		
		return stats;
	}
}
