package pt.up.fe.specs.clcuda;

import java.util.ArrayList;
import java.util.List;

public class KernelStats {
	public String name;
	public String launcherSymbolName;
	public List<ArgumentStats> args = new ArrayList<>();
}
