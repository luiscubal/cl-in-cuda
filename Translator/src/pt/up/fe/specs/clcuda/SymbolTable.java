package pt.up.fe.specs.clcuda;

import java.util.HashMap;
import java.util.Map;

import pt.up.fe.specs.clava.ast.extra.TranslationUnit;

public class SymbolTable {
	private SymbolTable parent;
	public TranslationUnit translationUnit;
	public Map<String, String> symbols = new HashMap<>();
	public Map<String, String> typeSymbols = new HashMap<>();
	public Map<String, Map<String, String>> structFields = new HashMap<>();
	
	public SymbolTable(TranslationUnit translationUnit) {
		this.parent = null;
		this.translationUnit = translationUnit;
	}
	
	public SymbolTable(SymbolTable parent) {
		this.parent = parent;
		this.translationUnit = parent.getTranslationUnit();
	}
	
	public void addSymbol(String name, String mangledName) {
		symbols.put(name, mangledName);
	}
	
	public Map<String, String> addTypeSymbol(String name, String mangledName) {
		typeSymbols.put(name, mangledName);
		
		Map<String, String> fields = new HashMap<>();
		structFields.put(name, fields);
		
		return fields;
	}
	
	public String getMangledName(String name) {
		String mangledName = symbols.get(name);
		if (mangledName != null) {
			return mangledName;
		}
		if (parent != null) {
			return parent.getMangledName(name);
		}
		return null;
	}

	public String getMangledTypeName(String name) {
		String mangledName = typeSymbols.get(name);
		if (mangledName != null) {
			return mangledName;
		}
		if (parent != null) {
			return parent.getMangledTypeName(name);
		}
		return null;
	}

	public String getMangledFieldName(String typeName, String memberName) {
		Map<String, String> mangledNames = structFields.get(typeName);
		if (mangledNames != null) {
			return mangledNames.get(memberName);
		}
		if (parent != null) {
			return parent.getMangledFieldName(typeName, memberName);
		}
		return null;
	}
	
	public TranslationUnit getTranslationUnit() {
		return translationUnit;
	}
}
