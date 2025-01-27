package pt.up.fe.specs.clcuda;

import pt.up.fe.specs.clava.ast.extra.TranslationUnit;
import pt.up.fe.specs.clava.ast.type.BuiltinType;

public class ArgumentStats {
	public ArgumentType type;
	public boolean isFloat;
	public int size;
	
	public static ArgumentStats fromGlobalPtr() {
		ArgumentStats stats = new ArgumentStats();
		stats.type = ArgumentType.GLOBAL_PTR;
		
		return stats;
	}
	
	public static ArgumentStats fromLocalPtr() {
		ArgumentStats stats = new ArgumentStats();
		stats.type = ArgumentType.LOCAL_PTR;
		
		return stats;
	}

	public static ArgumentStats fromScalar(BuiltinType type, TranslationUnit unit) {
		ArgumentStats stats = new ArgumentStats();
		stats.type = ArgumentType.SCALAR;
		stats.size = TypeUtils.getTypeSize(type, unit);
		
		return stats;
	}
}
