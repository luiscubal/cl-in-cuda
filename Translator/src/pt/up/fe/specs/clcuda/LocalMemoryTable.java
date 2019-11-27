package pt.up.fe.specs.clcuda;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import pt.up.fe.specs.clava.ast.type.Type;

public class LocalMemoryTable implements Iterable<LocalMemoryTable.TableEntry> {
	public static class TableEntry {
		String paramName;
		String offsetName;
		Type underlyingType;
		Type fullType;
		
		TableEntry(String paramName, String offsetName, Type fullType, Type underlyingType) {
			this.paramName = paramName;
			this.offsetName = offsetName;
			this.fullType = fullType;
			this.underlyingType = underlyingType;
		}
	}
	private List<TableEntry> entries = new ArrayList<>();
	
	public void add(String paramName, String offsetName, Type fullType, Type underlyingType) {
		entries.add(new TableEntry(paramName, offsetName, fullType, underlyingType));
	}
	
	public boolean isEmpty() {
		return entries.isEmpty();
	}

	public TableEntry get(int i) {
		return entries.get(i);
	}
	
	@Override
	public Iterator<TableEntry> iterator() {
		return entries.iterator();
	}
}
