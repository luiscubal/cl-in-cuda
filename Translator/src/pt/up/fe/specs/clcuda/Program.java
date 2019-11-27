package pt.up.fe.specs.clcuda;

import java.io.File;

import pt.up.fe.specs.util.SpecsSystem;

public class Program {
	public static void main(String[] args) {
		SpecsSystem.programStandardInit();
		
		ProgramResult result = new CLCuda().translate(new File(args[0]));
		System.out.println(result.cuda);
		System.out.println(result.toml);
	}
}
