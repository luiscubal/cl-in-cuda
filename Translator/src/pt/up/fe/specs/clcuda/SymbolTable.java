package pt.up.fe.specs.clcuda;

import java.util.HashMap;
import java.util.Map;

public class SymbolTable {
	private SymbolTable parent;
	public Map<String, String> symbols;
	public Map<String, String> typeSymbols;
	public Map<String, Map<String, String>> structFields;
	
	public SymbolTable(SymbolTable parent) {
		this.parent = parent;
		this.symbols = new HashMap<>();
		this.typeSymbols = new HashMap<>();
		this.structFields = new HashMap<>();
	}
	
	public void addSymbol(String name, String mangledName) {
		symbols.put(name, mangledName);
	}
	
	public Map<String, String> addStructSymbol(String name, String mangledName) {
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
}
