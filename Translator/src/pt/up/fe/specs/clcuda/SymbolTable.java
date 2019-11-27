package pt.up.fe.specs.clcuda;

import java.util.HashMap;
import java.util.Map;

public class SymbolTable {
	private SymbolTable parent;
	public Map<String, String> symbols;
	
	public SymbolTable(SymbolTable parent) {
		this.parent = parent;
		this.symbols = new HashMap<>();
	}
	
	public void addSymbol(String name, String mangledName) {
		symbols.put(name, mangledName);
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
}
