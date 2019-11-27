package pt.up.fe.specs.clcuda;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

import pt.up.fe.specs.util.SpecsSystem;

public class Program {
	public static void main(String[] args) throws IOException {
		SpecsSystem.programStandardInit();
		
		File openClFile = new File(args[0]);
		ProgramResult result = new CLCuda().translate(openClFile);
		
		File cudaFile = new File(openClFile.getParentFile(), openClFile.getName() + ".cu");
		File tomlFile = new File(openClFile.getParentFile(), openClFile.getName() + ".toml");
		try (PrintWriter out = new PrintWriter(cudaFile)) {
		    out.println(result.cuda);
		}
		try (PrintWriter out = new PrintWriter(tomlFile)) {
		    out.println(result.toml);
		}
	}
}
