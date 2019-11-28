package pt.up.fe.specs.clcuda.tests;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;

public class TestUtils {
	public static URL getTestResourceUrl(String resource) throws IOException {
		URL url = CLCudaTests.class.getClassLoader().getResource(resource);
		if (url == null) {
			throw new IOException("Could not find resource " + resource);
		}
		return url;
	}
	
	public static URI getTestResource(String resource) throws IOException, URISyntaxException {
		return getTestResourceUrl(resource).toURI();
	}
}
