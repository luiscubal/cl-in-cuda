package pt.up.fe.specs.clcuda.tests;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import pt.up.fe.specs.clcuda.CLCuda;
import pt.up.fe.specs.clcuda.ProgramResult;
import pt.up.fe.specs.lang.SpecsPlatforms;
import pt.up.fe.specs.util.SpecsSystem;

public class ExecutionTests {
    private static final String BASE_PATH = "pt/up/fe/specs/clcuda/tests/execution/";

    public ExecutionTests() {
        SpecsSystem.programStandardInit();
    }

    public void runTest(String testName, List<File> cFiles, List<File> clFiles, List<String> extraFlags)
            throws IOException {
        File containerFile = new File("").getAbsoluteFile().getParentFile();
        File runtimeFile = new File(containerFile, "Runtime");
        Path tempDir = Files.createTempDirectory(testName);
        System.out.println("Temp dir: " + tempDir);

        List<File> generatedCudaFiles = new ArrayList<>();
        for (File clFile : clFiles) {
            ProgramResult result = new CLCuda().translate(clFile);
            File cudaFile = new File(tempDir.toFile(), clFile.getName() + ".cu");
            try (PrintWriter writer = new PrintWriter(cudaFile)) {
                writer.println(result.cuda);
            }
            generatedCudaFiles.add(cudaFile);

            File tomlFile = new File(tempDir.toFile(), clFile.getName() + ".toml");
            try (PrintWriter writer = new PrintWriter(tomlFile)) {
                writer.println(result.toml);
            }
        }

        List<String> command = new ArrayList<>();
        command.add("nvcc");
        for (File file : cFiles) {
            command.add(file.getAbsolutePath());
        }
        for (File file : generatedCudaFiles) {
            command.add(file.getAbsolutePath());
        }

        for (File file : runtimeFile.listFiles()) {
            if (file.getName().endsWith(".cpp") || file.getName().endsWith(".cu")) {
                command.add(file.getAbsolutePath());
            }
        }

        command.add("-lcuda");
        command.add("-lcudart");
        command.add("-I");
        command.add(runtimeFile.getAbsolutePath());
        command.add("-o");
        String output = SpecsPlatforms.isWindows() ? "app.exe" : "app";
        command.add(output);
        command.addAll(extraFlags);

        int result = SpecsSystem.run(command, tempDir.toFile());
        Assert.assertEquals(0, result);

        File resultExe = new File(tempDir.toFile(), output);

        System.out.println("Running " + resultExe);
        result = SpecsSystem.run(Arrays.asList(resultExe.getAbsolutePath()), tempDir.toFile());
        Assert.assertEquals(0, result);
    }

    @Test
    public void testVectoradd() throws IOException, URISyntaxException {
        runTest("vectoradd",
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "vectoradd/vectoradd.c"))),
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "vectoradd/vectoradd.cl"))),
                Collections.emptyList());
    }

    @Test
    public void testMyGemm2() throws IOException, URISyntaxException {
        runTest("mygemm2",
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "cnugteren/mygemm2.c"))),
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "cnugteren/mygemm2.cl"))),
                Collections.emptyList());
    }

    @Test
    public void testReduceAddFloatLocalSize32() throws IOException, URISyntaxException {
        runTest("reduce_add_float",
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "reduce_add_float/reduce_add_float.c"))),
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "reduce_add_float/reduce_add_float.cl"))),
                Arrays.asList("-DLOCAL_SIZE=32"));
    }

    @Test
    public void testReduceAddFloatLocalSize128() throws IOException, URISyntaxException {
        runTest("reduce_add_float",
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "reduce_add_float/reduce_add_float.c"))),
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "reduce_add_float/reduce_add_float.cl"))),
                Arrays.asList("-DLOCAL_SIZE=128"));
    }

    @Test
    public void testReduceAddFloatLocalSize1024() throws IOException, URISyntaxException {
        runTest("reduce_add_float",
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "reduce_add_float/reduce_add_float.c"))),
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "reduce_add_float/reduce_add_float.cl"))),
                Arrays.asList("-DLOCAL_SIZE=1024"));
    }

    @Test
    public void testFillBuffer() throws IOException, URISyntaxException {
        runTest("fillbuffer",
                Arrays.asList(
                        new File(TestUtils.getTestResource(BASE_PATH + "fillbuffer/fillbuffer.c"))),
                Collections.emptyList(),
                Collections.emptyList());
    }
}
