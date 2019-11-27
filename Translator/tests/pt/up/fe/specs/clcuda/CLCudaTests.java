package pt.up.fe.specs.clcuda;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;

import org.junit.Assert;
import org.junit.Test;

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
}
