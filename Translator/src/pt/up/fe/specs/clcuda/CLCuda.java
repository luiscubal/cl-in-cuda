package pt.up.fe.specs.clcuda;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import pt.up.fe.specs.clang.codeparser.CodeParser;
import pt.up.fe.specs.clava.ClavaNode;
import pt.up.fe.specs.clava.ast.attr.OpenCLKernelAttr;
import pt.up.fe.specs.clava.ast.decl.Decl;
import pt.up.fe.specs.clava.ast.decl.FunctionDecl;
import pt.up.fe.specs.clava.ast.decl.ParmVarDecl;
import pt.up.fe.specs.clava.ast.decl.VarDecl;
import pt.up.fe.specs.clava.ast.expr.ArraySubscriptExpr;
import pt.up.fe.specs.clava.ast.expr.BinaryOperator;
import pt.up.fe.specs.clava.ast.expr.CallExpr;
import pt.up.fe.specs.clava.ast.expr.DeclRefExpr;
import pt.up.fe.specs.clava.ast.expr.Expr;
import pt.up.fe.specs.clava.ast.expr.FloatingLiteral;
import pt.up.fe.specs.clava.ast.expr.IntegerLiteral;
import pt.up.fe.specs.clava.ast.expr.enums.BinaryOperatorKind;
import pt.up.fe.specs.clava.ast.extra.App;
import pt.up.fe.specs.clava.ast.extra.TranslationUnit;
import pt.up.fe.specs.clava.ast.stmt.CompoundStmt;
import pt.up.fe.specs.clava.ast.stmt.DeclStmt;
import pt.up.fe.specs.clava.ast.stmt.ExprStmt;
import pt.up.fe.specs.clava.ast.stmt.IfStmt;
import pt.up.fe.specs.clava.ast.stmt.NullStmt;
import pt.up.fe.specs.clava.ast.stmt.Stmt;
import pt.up.fe.specs.clava.ast.type.BuiltinType;
import pt.up.fe.specs.clava.ast.type.PointerType;
import pt.up.fe.specs.clava.ast.type.QualType;
import pt.up.fe.specs.clava.ast.type.Type;
import pt.up.fe.specs.clava.ast.type.TypedefType;
import pt.up.fe.specs.clava.ast.type.enums.AddressSpaceQualifierV2;
import pt.up.fe.specs.util.exceptions.NotImplementedException;

public class CLCuda {
	public ProgramResult translate(File file) {
		App app = CodeParser.newInstance().parse(Arrays.asList(file), Arrays.asList("-Wall"));
		return generateCode(app);
	}
	
	public ProgramResult generateCode(App app) {
		ProgramStats programStats = new ProgramStats();
		programStats.buildOpts = "";
		String cuda = generateCuda(app, programStats);
		return new ProgramResult(cuda, programStats.toToml());
	}
	
	private String generateCuda(App app, ProgramStats stats) {
		StringBuilder builder = new StringBuilder();
		builder.append("#include \"cl_device_assist.cuh\"\n");
		builder.append("#include \"cl_interface_shared.h\"\n\n");
		
		assert app.getNumChildren() == 1 : "Expected single translation unit";
		
		TranslationUnit unit = (TranslationUnit) app.getChild(0);
		SymbolTable rootTable = new SymbolTable(null);
		
		for (ClavaNode child : unit.getChildren()) {
			if (child instanceof FunctionDecl) {
				generateCodeForFunctionDecl((FunctionDecl) child, unit, builder, rootTable, stats);
			} else {
				throw new NotImplementedException(child.getClass());
			}
		}
		
		return builder.toString();
	}
	
	private void generateCodeForFunctionDecl(FunctionDecl func, TranslationUnit unit, StringBuilder builder, SymbolTable parentTable, ProgramStats stats) {
		String declName = func.getDeclName();
		String symbolName = "clcuda_func_" + declName;
		parentTable.addSymbol(declName, symbolName);
		SymbolTable funcTable = new SymbolTable(parentTable);
		
		boolean isKernel = false;
		for (ClavaNode nodeField : func.getNodeFields()) {
			if (nodeField instanceof OpenCLKernelAttr) {
				isKernel = true;
			}
		}
		
		builder.append(isKernel ? "__global__ " : "__device__ ");
		generateCodeForType(func.getReturnType(), builder);
		builder.append(" ");
		builder.append(symbolName);
		builder.append("(");
		for (ParmVarDecl param : func.getParameters()) {
			String paramName = param.getDeclName();
			funcTable.addSymbol(paramName, "var_" + paramName);
			generateVarDecl(param.getType(), "var_" + paramName, builder);
			builder.append(", ");
		}
		builder.append("CommonKernelData data");
		builder.append(")\n{\n");
		builder.append("\tif (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;\n");
		builder.append("\tif (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;\n");
		builder.append("\tif (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;\n\n");
		
		if (func.getBody().isPresent()) {
			CompoundStmt stmt = func.getBody().get();
			buildBody(stmt, builder, funcTable, "\t");
		}
		
		builder.append("}\n\n");
		
		if (isKernel) {
			String launcherName = "clcuda_launcher_" + declName;
			KernelStats kernelStats = new KernelStats();
			stats.kernels.add(kernelStats);
			kernelStats.name = declName;
			kernelStats.launcherSymbolName = launcherName;
			builder.append("KERNEL_LAUNCHER void ");
			builder.append(launcherName);
			builder.append("(struct _cl_kernel *desc)\n");
			builder.append("{\n");
			builder.append("\tdim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);\n");
			builder.append("\tdim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);\n");
			builder.append("\n\t");
			builder.append(symbolName);
			builder.append("<<<num_grids, local_size>>>(\n");
			List<ParmVarDecl> parameters = func.getParameters();
			for (int i = 0; i < parameters.size(); i++) {
				ParmVarDecl param = parameters.get(i);
				Type paramType = param.getType();
				if (paramType instanceof QualType) {
					paramType = ((QualType) paramType).getUnqualifiedType();
				}
				if (paramType instanceof PointerType) {
					Type pointeeType = ((PointerType) paramType).getPointeeType();
					if (pointeeType.get(QualType.ADDRESS_SPACE_QUALIFIER) == AddressSpaceQualifierV2.GLOBAL) {
						kernelStats.args.add(ArgumentStats.fromGlobalPtr());
						builder.append("\t\t(");
						generateCodeForType(paramType, builder);
						builder.append(") desc->arg_data[");
						builder.append(i);
						builder.append("],\n");
					} else {
						throw new NotImplementedException(pointeeType.getClass());
					}
					continue;
				}
				if (paramType instanceof BuiltinType) {
					kernelStats.args.add(ArgumentStats.fromScalar((BuiltinType) paramType, unit));
					builder.append("\t\t*(");
					generateCodeForType(paramType, builder);
					builder.append("*) desc->arg_data[");
					builder.append(i);
					builder.append("],\n");
					continue;
				}
				throw new NotImplementedException(paramType.getClass());
			}
			builder.append("\t\tCommonThreadData(desc->totalX, desc->totalY, desc->totalZ)\n");
			builder.append("\t);\n");
			builder.append("}\n\n");
		}
	}
	
	private void buildStmt(Stmt stmt, StringBuilder builder, SymbolTable symTable, String indentation) {
		if (stmt instanceof DeclStmt) {
			DeclStmt declStmt = (DeclStmt) stmt;
			
			for (Decl decl : declStmt.getDecls()) {
				VarDecl varDecl = (VarDecl) decl;
				
				String name = varDecl.getDeclName();
				String mangledName = "var_" + name;
				symTable.addSymbol(name, mangledName);
				
				builder.append(indentation);
				generateVarDecl(varDecl.getType(), mangledName, builder);
				
				if (varDecl.getNumChildren() > 0) {
					builder.append(" = ");
					buildExpr((Expr) varDecl.getChild(0), symTable, builder);
				}
				builder.append(";\n");
			}
			return;
		}
		if (stmt instanceof ExprStmt) {
			builder.append(indentation);
			buildExpr(((ExprStmt) stmt).getExpr(), symTable, builder);
			builder.append(";\n");
			return;
		}
		if (stmt instanceof IfStmt) {
			IfStmt ifStmt = (IfStmt) stmt;
			Stmt thenCase = (Stmt)ifStmt.getChild(2);
			Stmt elseCase = (Stmt)ifStmt.getChild(3);
			
			builder.append(indentation);
			builder.append("if (");
			buildExpr(ifStmt.getCondition(), symTable, builder);
			builder.append(")\n");
			if (thenCase instanceof NullStmt) {
				builder.append(indentation);
				builder.append("{\n");
				builder.append(indentation);
				builder.append("}\n");
			} else {
				buildStmt(thenCase, builder, symTable, indentation);
			}
			if (!(elseCase instanceof NullStmt)) {
				builder.append("else\n");
				buildStmt(elseCase, builder, symTable, indentation);
			}
			return;
		}
		if (stmt instanceof CompoundStmt) {
			CompoundStmt compoundStmt = (CompoundStmt) stmt;
			builder.append(indentation);
			builder.append("{\n");
			buildBody(compoundStmt, builder, new SymbolTable(symTable), indentation + "\t");
			builder.append(indentation);
			builder.append("}\n");
			return;
		}
		throw new NotImplementedException(stmt.getClass());
	}
	
	private void buildBody(CompoundStmt block, StringBuilder builder, SymbolTable symTable, String indentation) {
		for (Stmt stmt : block.getChildren(Stmt.class)) {
			buildStmt(stmt, builder, symTable, indentation);
		}
	}
	
	private void buildExpr(Expr expr, SymbolTable symTable, StringBuilder builder) {
		if (expr instanceof IntegerLiteral) {
			builder.append(expr.getCode());
			return;
		}
		if (expr instanceof CallExpr) {
			CallExpr callExpr = (CallExpr) expr;
			mayParenthiseCallee(callExpr.getCallee(), symTable, builder);
			builder.append("(");
			for (Expr arg : callExpr.getArgs()) {
				buildExpr(arg, symTable, builder);
				builder.append(", ");
			}
			builder.append("data)");
			return;
		}
		if (expr instanceof DeclRefExpr) {
			DeclRefExpr declRefExpr = (DeclRefExpr) expr;
			
			String name = declRefExpr.getName();
			String mangledName = symTable.getMangledName(name);
			if (mangledName == null) {
				checkSafeName(name);
				builder.append(name);
			} else {
				builder.append(mangledName);
			}
			return;
		}
		if (expr instanceof BinaryOperator) {
			BinaryOperator binaryOperator = (BinaryOperator) expr;
			Expr left = (Expr) binaryOperator.getChild(0);
			Expr right = (Expr) binaryOperator.getChild(1);
			mayParenthiseOperand(left, binaryOperator.getOp(), symTable, builder);
			builder.append(" ");
			builder.append(binaryOperator.getOp().getOpString());
			builder.append(" ");
			mayParenthiseOperand(right, binaryOperator.getOp(), symTable, builder);
			return;
		}
		if (expr instanceof ArraySubscriptExpr) {
			ArraySubscriptExpr arraySubscriptExpr = (ArraySubscriptExpr) expr;
			mayParenthiseReferencedArray((Expr)arraySubscriptExpr.getChild(0), symTable, builder);
			builder.append("[");
			buildExpr((Expr) arraySubscriptExpr.getChild(1), symTable, builder);
			builder.append("]");
			return;
		}
		if (expr instanceof FloatingLiteral) {
			builder.append(expr.getCode());
			return;
		}
		System.out.println(expr);
	}
	
	private void mayParenthiseOperand(Expr operand, BinaryOperatorKind op, SymbolTable symTable, StringBuilder builder) {
		boolean shouldParenthise = !(isBasicUnit(operand) || operand instanceof CallExpr);
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}
	
	private void mayParenthiseCallee(Expr operand, SymbolTable symTable, StringBuilder builder) {
		boolean shouldParenthise = !(isBasicUnit(operand) || operand instanceof CallExpr);
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}
	
	private void mayParenthiseReferencedArray(Expr operand, SymbolTable symTable, StringBuilder builder) {
		boolean shouldParenthise = !(isBasicUnit(operand) || operand instanceof CallExpr);
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}
	
	private boolean isBasicUnit(Expr expr) {
		return expr instanceof DeclRefExpr ||
				expr instanceof FloatingLiteral;
	}
	
	private void checkSafeName(String name) {
		switch (name) {
		case "get_global_id":
			return;
		default:
			throw new RuntimeException("Unknown name: " + name);
		}
	}

	private void generateVarDecl(Type type, String paramName, StringBuilder builder) {
		if (type instanceof QualType) {
			generateVarDecl(((QualType) type).getUnqualifiedType(), paramName, builder);
			return;
		}
		if (type instanceof PointerType) {
			PointerType pointerType = (PointerType) type;
			generateCodeForType(pointerType.getPointeeType(), builder);
			builder.append(" *");
			builder.append(paramName);
			return;
		}
		
		generateCodeForType(type, builder);
		builder.append(" ");
		builder.append(paramName);
	}

	private void generateCodeForType(Type type, StringBuilder builder) {
		if (type instanceof QualType) {
			QualType qualType = (QualType) type;
			generateCodeForType(qualType.getUnqualifiedType(), builder);
			return;
		}
		if (type instanceof PointerType) {
			PointerType pointerType = (PointerType) type;
			generateCodeForType(pointerType.getPointeeType(), builder);
			builder.append("*");
			return;
		}
		if (type instanceof TypedefType) {
			if (type.getCode().equals("size_t")) {
				builder.append(type.getCode());
				return;
			}
		}
		// STUB
		builder.append(type.getCode());
	}
}
