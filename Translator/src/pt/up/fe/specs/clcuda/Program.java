package pt.up.fe.specs.clcuda;

import java.io.File;

public class Program {
	public static void main(String[] args) {
		new CLCuda().translate(new File(args[0]));
	}
}
