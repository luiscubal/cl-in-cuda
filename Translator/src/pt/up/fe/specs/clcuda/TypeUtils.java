package pt.up.fe.specs.clcuda;

import pt.up.fe.specs.clava.ast.extra.TranslationUnit;
import pt.up.fe.specs.clava.ast.type.BuiltinType;

public class TypeUtils {
	public static int getTypeSize(BuiltinType type, TranslationUnit translationUnit) {
		return type.get(BuiltinType.KIND).getBitwidth(translationUnit.get(TranslationUnit.LANGUAGE)) / 8;
	}
}
