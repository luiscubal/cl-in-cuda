package pt.up.fe.specs.clcuda;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;

import org.junit.Assert;
import org.junit.Test;

import pt.up.fe.specs.util.SpecsSystem;

public class CLCudaTests {
	private static URL getTestResourceUrl(String resource) {
		return CLCudaTests.class.getClassLoader().getResource("pt/up/fe/specs/clcuda/" + resource);
	}
	
	private static URI getTestResource(String resource) throws URISyntaxException {
		return getTestResourceUrl(resource).toURI();
	}
	
	private static String readTestResource(String resource) throws URISyntaxException, IOException {
		try (InputStream in = getTestResourceUrl(resource).openStream()) {
			byte[] bytes = in.readAllBytes();
			return new String(bytes, "utf-8").replace("\r\n", "\n");
		}
	}
	
	private static void validateProgram(ProgramResult result, String expectedCuda, String expectedToml) {
		Assert.assertEquals(expectedCuda, result.cuda);
		Assert.assertEquals(expectedToml, result.toml);
	}
	
	public CLCudaTests() {
		SpecsSystem.programStandardInit();
	}
	
	@Test
	public void testVectorAdd() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("vectoradd.cl"))),
				readTestResource("vectoradd.cu"),
				readTestResource("vectoradd.toml"));
	}
	
	@Test
	public void testManualBound() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("manualbound.cl"))),
				readTestResource("manualbound.cu"),
				readTestResource("manualbound.toml"));
	}
	
	@Test
	public void testBranches() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("branches.cl"))),
				readTestResource("branches.cu"),
				readTestResource("branches.toml"));
	}
	
	@Test
	public void testWhileLoops() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("while_loop.cl"))),
				readTestResource("while_loop.cu"),
				readTestResource("while_loop.toml"));
	}
	
	@Test
	public void testForLoops() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("for_loops.cl"))),
				readTestResource("for_loops.cu"),
				readTestResource("for_loops.toml"));
	}
	
	@Test
	public void testStructs() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("structs.cl"))),
				readTestResource("structs.cu"),
				readTestResource("structs.toml"));
	}
	
	@Test
	public void testModifiers() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("modifiers.cl"))),
				readTestResource("modifiers.cu"),
				readTestResource("modifiers.toml"));
	}
	
	@Test
	public void testComment() throws URISyntaxException, IOException {
		validateProgram(
				new CLCuda().translate(new File(getTestResource("comment.cl"))),
				readTestResource("comment.cu"),
				readTestResource("comment.toml"));
	}
}
