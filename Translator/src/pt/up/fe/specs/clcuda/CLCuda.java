package pt.up.fe.specs.clcuda;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.jaxen.expr.UnaryExpr;

import pt.up.fe.specs.clang.codeparser.CodeParser;
import pt.up.fe.specs.clava.ClavaNode;
import pt.up.fe.specs.clava.ast.attr.OpenCLKernelAttr;
import pt.up.fe.specs.clava.ast.comment.InlineComment;
import pt.up.fe.specs.clava.ast.comment.MultiLineComment;
import pt.up.fe.specs.clava.ast.decl.Decl;
import pt.up.fe.specs.clava.ast.decl.FieldDecl;
import pt.up.fe.specs.clava.ast.decl.FunctionDecl;
import pt.up.fe.specs.clava.ast.decl.ParmVarDecl;
import pt.up.fe.specs.clava.ast.decl.RecordDecl;
import pt.up.fe.specs.clava.ast.decl.VarDecl;
import pt.up.fe.specs.clava.ast.expr.ArraySubscriptExpr;
import pt.up.fe.specs.clava.ast.expr.BinaryOperator;
import pt.up.fe.specs.clava.ast.expr.CallExpr;
import pt.up.fe.specs.clava.ast.expr.DeclRefExpr;
import pt.up.fe.specs.clava.ast.expr.Expr;
import pt.up.fe.specs.clava.ast.expr.FloatingLiteral;
import pt.up.fe.specs.clava.ast.expr.IntegerLiteral;
import pt.up.fe.specs.clava.ast.expr.MemberExpr;
import pt.up.fe.specs.clava.ast.expr.ParenExpr;
import pt.up.fe.specs.clava.ast.expr.UnaryOperator;
import pt.up.fe.specs.clava.ast.expr.enums.BinaryOperatorKind;
import pt.up.fe.specs.clava.ast.expr.enums.UnaryOperatorKind;
import pt.up.fe.specs.clava.ast.extra.App;
import pt.up.fe.specs.clava.ast.extra.TranslationUnit;
import pt.up.fe.specs.clava.ast.stmt.BreakStmt;
import pt.up.fe.specs.clava.ast.stmt.CompoundStmt;
import pt.up.fe.specs.clava.ast.stmt.ContinueStmt;
import pt.up.fe.specs.clava.ast.stmt.DeclStmt;
import pt.up.fe.specs.clava.ast.stmt.ExprStmt;
import pt.up.fe.specs.clava.ast.stmt.ForStmt;
import pt.up.fe.specs.clava.ast.stmt.IfStmt;
import pt.up.fe.specs.clava.ast.stmt.NullStmt;
import pt.up.fe.specs.clava.ast.stmt.ReturnStmt;
import pt.up.fe.specs.clava.ast.stmt.Stmt;
import pt.up.fe.specs.clava.ast.stmt.WhileStmt;
import pt.up.fe.specs.clava.ast.stmt.WrapperStmt;
import pt.up.fe.specs.clava.ast.type.BuiltinType;
import pt.up.fe.specs.clava.ast.type.ElaboratedType;
import pt.up.fe.specs.clava.ast.type.PointerType;
import pt.up.fe.specs.clava.ast.type.QualType;
import pt.up.fe.specs.clava.ast.type.RecordType;
import pt.up.fe.specs.clava.ast.type.Type;
import pt.up.fe.specs.clava.ast.type.TypedefType;
import pt.up.fe.specs.clava.ast.type.enums.AddressSpaceQualifierV2;
import pt.up.fe.specs.clava.ast.type.enums.C99Qualifier;
import pt.up.fe.specs.clava.language.TagKind;
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
				continue;
			}
			if (child instanceof RecordDecl) {
				RecordDecl recordDecl = (RecordDecl) child;
				switch (recordDecl.getTagKind()) {
				case STRUCT:
					generateCodeForStructDecl(recordDecl, unit, builder, rootTable, stats);
					break;
				default:
					throw new NotImplementedException(recordDecl.getTagKind().toString());
				}
				continue;
			}
			if (child instanceof InlineComment || child instanceof MultiLineComment) {
				builder.append(child.getCode().replace("\r\n", "\n"));
				builder.append("\n");
				continue;
			}
			throw new NotImplementedException(child.getClass());
		}
		
		return builder.toString();
	}
	
	private void generateCodeForStructDecl(RecordDecl struct, TranslationUnit unit, StringBuilder builder, SymbolTable parentTable, ProgramStats stats) {
		String declName = struct.getDeclName();
		String symbolName = "clcuda_type_" + declName;
		Map<String, String> fieldTable = parentTable.addStructSymbol(declName, symbolName);
		
		builder.append("struct ");
		builder.append(symbolName);
		builder.append("\n{\n");
		
		for (FieldDecl field : struct.getFields()) {
			String fieldName = field.getDeclName();
			String fieldSymbolName = "field_" + fieldName;
			fieldTable.put(fieldName, fieldSymbolName);
			
			builder.append("\t");
			generateVarDecl(field.getType(), fieldSymbolName, parentTable, builder);
			builder.append(";\n");
		}
		
		builder.append("};\n\n");
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
		generateCodeForType(func.getReturnType(), funcTable, builder);
		builder.append(" ");
		builder.append(symbolName);
		builder.append("(");
		
		LocalMemoryTable localMemoryTable = new LocalMemoryTable();
		
		for (ParmVarDecl param : func.getParameters()) {
			String paramName = param.getDeclName();
			funcTable.addSymbol(paramName, "var_" + paramName);
			
			Type paramType = param.getType();
			Type processedType = paramType;
			if (paramType instanceof QualType) {
				processedType = ((QualType) paramType).getUnqualifiedType();
			}
			
			if (processedType instanceof PointerType) {
				Type pointeeType = ((PointerType) processedType).getPointeeType();
				if (isKernel && pointeeType.get(QualType.ADDRESS_SPACE_QUALIFIER) == AddressSpaceQualifierV2.LOCAL) {
					String offsetName = "clcuda_offset_" + paramName;
					builder.append("size_t ");
					builder.append(offsetName);
					builder.append(", ");
					localMemoryTable.add(paramName, offsetName, paramType, ((QualType) pointeeType).getUnqualifiedType());
					continue;
				}
			}
			generateVarDecl(paramType, "var_" + paramName, funcTable, builder);
			builder.append(", ");
		}
		builder.append("CommonKernelData data");
		builder.append(")\n{\n");
		if (isKernel) {
			builder.append("\tif (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;\n");
			builder.append("\tif (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;\n");
			builder.append("\tif (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;\n\t\n");
			
			if (!localMemoryTable.isEmpty()) {
				builder.append("\textern __shared__ char local_mem[];\n");
				for (LocalMemoryTable.TableEntry entry : localMemoryTable) {
					builder.append("\t");
					generateVarDecl(entry.fullType, "var_" + entry.paramName, funcTable, builder);
					builder.append(" = (");
					generateCodeForType(entry.fullType, funcTable, builder);
					builder.append(") (local_mem + ");
					builder.append(entry.offsetName);
					builder.append(");\n");
				}
				builder.append("\t\n");
			}
		}
		
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
			builder.append("\t\n");
			
			StringBuilder localMemBuilder = new StringBuilder();
			if (!localMemoryTable.isEmpty()) {
				builder.append("\tsize_t local_mem_size = 0;\n");
			}
			
			StringBuilder kernelCallBuilder = new StringBuilder();
			kernelCallBuilder.append("\t");
			kernelCallBuilder.append(symbolName);
			kernelCallBuilder.append("<<<num_grids, local_size");
			if (!localMemoryTable.isEmpty()) {
				kernelCallBuilder.append(", local_mem_size");
			}
			kernelCallBuilder.append(">>>(\n");
			int localMemoryTableIndex = 0;
			List<ParmVarDecl> parameters = func.getParameters();
			for (int argumentIndex = 0; argumentIndex < parameters.size(); argumentIndex++) {
				ParmVarDecl param = parameters.get(argumentIndex);
				Type paramType = param.getType();
				Type processedType = paramType;
				if (processedType instanceof QualType) {
					QualType qualType = (QualType) processedType;
					processedType = qualType.getUnqualifiedType();
				}
				if (processedType instanceof TypedefType) {
					processedType = ((TypedefType) processedType).desugar();
				}
				if (processedType instanceof PointerType) {
					Type pointeeType = ((PointerType) processedType).getPointeeType();
					if (pointeeType.get(QualType.ADDRESS_SPACE_QUALIFIER) == AddressSpaceQualifierV2.GLOBAL) {
						kernelStats.args.add(ArgumentStats.fromGlobalPtr());
						kernelCallBuilder.append("\t\t(");
						generateCodeForType(paramType, funcTable, kernelCallBuilder);
						kernelCallBuilder.append(") desc->arg_data[");
						kernelCallBuilder.append(argumentIndex);
						kernelCallBuilder.append("],\n");
					} else {
						LocalMemoryTable.TableEntry entry = localMemoryTable.get(localMemoryTableIndex);
						kernelStats.args.add(ArgumentStats.fromLocalPtr());
						kernelCallBuilder.append("\t\t");
						kernelCallBuilder.append(entry.offsetName);
						kernelCallBuilder.append(",\n");
						
						if (localMemoryTableIndex > 0) {
							int size = entry.underlyingType.get(BuiltinType.KIND).getBitwidth(unit.get(TranslationUnit.LANGUAGE)) / 8;
							localMemBuilder.append("\tlocal_mem_size = ((local_mem_size + ");
							localMemBuilder.append(size - 1);
							localMemBuilder.append(") / ");
							localMemBuilder.append(size);
							localMemBuilder.append(") * ");
							localMemBuilder.append(size);
							localMemBuilder.append(";\n");
						}
						
						localMemBuilder.append("\tsize_t ");
						localMemBuilder.append(entry.offsetName);
						localMemBuilder.append(" = local_mem_size;\n");
						localMemBuilder.append("\tlocal_mem_size += desc->arg_data[");
						localMemBuilder.append(argumentIndex);
						localMemBuilder.append("];\n");
						
						localMemoryTableIndex++;
					}
					continue;
				}
				if (processedType instanceof BuiltinType) {
					kernelStats.args.add(ArgumentStats.fromScalar((BuiltinType) processedType, unit));
					kernelCallBuilder.append("\t\t*(");
					generateCodeForType(paramType, funcTable, kernelCallBuilder);
					kernelCallBuilder.append("*) desc->arg_data[");
					kernelCallBuilder.append(argumentIndex);
					kernelCallBuilder.append("],\n");
					
					continue;
				}
				System.out.println(processedType);
				throw new NotImplementedException(processedType.getClass());
			}
			kernelCallBuilder.append("\t\tCommonThreadData(desc->totalX, desc->totalY, desc->totalZ)\n");
			kernelCallBuilder.append("\t);");
			if (!localMemoryTable.isEmpty()) {
				builder.append(localMemBuilder);
				builder.append("\t\n");
			}
			builder.append(kernelCallBuilder);
			builder.append("\n");
			builder.append("}\n\n");
		}
	}
	
	private void buildUnseparatedStmt(Stmt stmt, StringBuilder builder, SymbolTable symTable, String indentation) {
		if (stmt instanceof DeclStmt) {
			DeclStmt declStmt = (DeclStmt) stmt;
			
			for (Decl decl : declStmt.getDecls()) {
				VarDecl varDecl = (VarDecl) decl;
				
				String name = varDecl.getDeclName();
				String mangledName = "var_" + name;
				symTable.addSymbol(name, mangledName);
				
				builder.append(indentation);
				generateVarDecl(varDecl.getType(), mangledName, symTable, builder);
				
				if (varDecl.getNumChildren() > 0) {
					builder.append(" = ");
					buildExpr((Expr) varDecl.getChild(0), symTable, builder);
				}
			}
			return;
		}
		if (stmt instanceof ExprStmt) {
			builder.append(indentation);
			buildExpr(((ExprStmt) stmt).getExpr(), symTable, builder);
			return;
		}

		throw new NotImplementedException(stmt.getClass());
	}
	
	private void buildStmt(Stmt stmt, StringBuilder builder, SymbolTable symTable, String indentation) {
		if (stmt instanceof WrapperStmt) {
			// Risky: ignore #pragmas, at least for now
			builder.append(indentation);
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
				builder.append("}");
			} else {
				buildStmt(thenCase, builder, symTable, indentation);
			}
			if (!(elseCase instanceof NullStmt)) {
				builder.append(" else\n");
				buildStmt(elseCase, builder, symTable, indentation);
			}
			return;
		}
		if (stmt instanceof WhileStmt) {
			WhileStmt ifStmt = (WhileStmt) stmt;
			Stmt thenCase = (Stmt)ifStmt.getChild(2);
			
			builder.append(indentation);
			builder.append("while (");
			buildExpr(ifStmt.getCondition(), symTable, builder);
			builder.append(")\n");
			if (thenCase instanceof NullStmt) {
				builder.append(indentation);
				builder.append("{\n");
				builder.append(indentation);
				builder.append("}");
			} else {
				buildStmt(thenCase, builder, symTable, indentation);
			}
			return;
		}
		if (stmt instanceof CompoundStmt) {
			CompoundStmt compoundStmt = (CompoundStmt) stmt;
			builder.append(indentation);
			builder.append("{\n");
			buildBody(compoundStmt, builder, new SymbolTable(symTable), indentation + "\t");
			builder.append(indentation);
			builder.append("}");
			return;
		}
		if (stmt instanceof ForStmt) {
			ForStmt forStmt = (ForStmt) stmt;
			builder.append(indentation);
			builder.append("for (");
			SymbolTable forTable = new SymbolTable(symTable);
			if (forStmt.getInit().isPresent()) {
				Stmt init = forStmt.getInit().get();
				buildUnseparatedStmt(init, builder, forTable, "");
			}
			builder.append("; ");
			if (forStmt.getCond().isPresent()) {
				Stmt cond = forStmt.getCond().get();
				buildUnseparatedStmt(cond, builder, forTable, "");
			}
			builder.append(";");
			if (forStmt.getInc().isPresent()) {
				Stmt inc = forStmt.getInc().get();
				builder.append(" ");
				buildUnseparatedStmt(inc, builder, forTable, "");
			}
			builder.append(")\n");
			buildStmt(forStmt.getBody(), builder, forTable, indentation);
			return;
		}
		if (stmt instanceof ReturnStmt) {
			builder.append(indentation);
			builder.append("return");
			if (stmt.getNumChildren() != 0) {
				builder.append(" ");
				buildExpr((Expr) stmt.getChild(0), symTable, builder);
			}
			builder.append(";");
			return;
		}
		if (stmt instanceof BreakStmt) {
			// FIXME: support labelled breaks?
			builder.append(indentation);
			builder.append("break;");
			return;
		}
		if (stmt instanceof ContinueStmt) {
			// FIXME: support labelled continues?
			builder.append(indentation);
			builder.append("continue;");
			return;
		}
		buildUnseparatedStmt(stmt, builder, symTable, indentation);
		builder.append(";");
	}
	
	private void buildBody(CompoundStmt block, StringBuilder builder, SymbolTable symTable, String indentation) {
		for (Stmt stmt : block.getChildren(Stmt.class)) {
			buildStmt(stmt, builder, symTable, indentation);
			builder.append("\n");
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
				builder.append("clcuda_builtin_");
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
		if (expr instanceof MemberExpr) {
			MemberExpr memberExpr = (MemberExpr) expr;
			mayParenthiseMemberBase(memberExpr.getBase(), symTable, builder);
			builder.append(memberExpr.isArrow() ? "->" : ".");
			
			builder.append(getMangledFieldName(memberExpr.getBase(), memberExpr.getMemberName(), symTable));
			return;
		}
		if (expr instanceof UnaryOperator) {
			UnaryOperator unaryOperator = (UnaryOperator) expr;
			Expr operand = (Expr) unaryOperator.getChild(0);
			UnaryOperatorKind kind = unaryOperator.getOp();
			if (kind == UnaryOperatorKind.PostDec || kind == UnaryOperatorKind.PostInc) {
				mayParenthiseUnaryOperand(operand, kind, symTable, builder);
				builder.append(kind.getCode());
			} else {
				builder.append(kind.getCode());
				mayParenthiseUnaryOperand(operand, kind, symTable, builder);
			}
			return;
		}
		if (expr instanceof ParenExpr) {
			buildExpr((Expr)expr.getChild(0), symTable, builder);
			return;
		}
		System.out.println(expr);
	}

	private Object getMangledFieldName(Expr base, String memberName, SymbolTable symTable) {
		Type type = base.getType();
		RecordType recordType = getUnderlyingRecordType(type);
		
		return symTable.getMangledFieldName(recordType.getDecl().getDeclName(), memberName);
	}

	private RecordType getUnderlyingRecordType(Type type) {
		if (type instanceof QualType) {
			return getUnderlyingRecordType(((QualType) type).getUnqualifiedType());
		}
		if (type instanceof RecordType) {
			return (RecordType) type;
		}
		if (type instanceof ElaboratedType) {
			return getUnderlyingRecordType(((ElaboratedType) type).getNamedType());
		}
		
		return null;
	}
	
	private void mayParenthiseUnaryOperand(Expr operand, UnaryOperatorKind kind, SymbolTable symTable,
			StringBuilder builder) {
		
		boolean shouldParenthise = !isSimpleUnit(operand);
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}

	private void mayParenthiseOperand(Expr operand, BinaryOperatorKind op, SymbolTable symTable, StringBuilder builder) {
		boolean shouldParenthise = !isSimpleUnit(operand);
		if (shouldParenthise) {
			if (operand instanceof BinaryOperator) {
				BinaryOperatorKind childOp = ((BinaryOperator) operand).getOp();
				if (childOp == op && isAssociative(op)) {
					shouldParenthise = false;
				}
			}
		}
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}
	
	private void mayParenthiseCallee(Expr operand, SymbolTable symTable, StringBuilder builder) {
		boolean shouldParenthise = !isSimpleUnit(operand);
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}
	
	private void mayParenthiseReferencedArray(Expr operand, SymbolTable symTable, StringBuilder builder) {
		boolean shouldParenthise = !isSimpleUnit(operand);
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}
	
	private void mayParenthiseMemberBase(Expr operand, SymbolTable symTable, StringBuilder builder) {
		boolean shouldParenthise = !isSimpleUnit(operand);
		if (shouldParenthise) {
			builder.append("(");
		}
		buildExpr(operand, symTable, builder);
		if (shouldParenthise) {
			builder.append(")");
		}
	}
	
	private boolean isAssociative(BinaryOperatorKind kind) {
		switch (kind) {
		case Add:
		case Sub:
		case Mul:
			return true;
		default:
			return false;
		}
	}
	
	private boolean isSimpleUnit(Expr expr) {
		return isAtomicUnit(expr) ||
				expr instanceof CallExpr ||
				expr instanceof MemberExpr ||
				expr instanceof ArraySubscriptExpr;
	}
	
	private boolean isAtomicUnit(Expr expr) {
		return expr instanceof DeclRefExpr ||
				expr instanceof FloatingLiteral ||
				expr instanceof IntegerLiteral;
	}
	
	private void checkSafeName(String name) {
		switch (name) {
		case "get_global_id":
		case "get_global_size":
		case "get_local_id":
		case "get_local_size":
		case "get_group_id":
		case "barrier":
		case "sqrt":
		case "exp":
		case "log":
		case "sin":
		case "cos":
		case "tan":
			return;
		default:
			throw new RuntimeException("Unknown name: " + name);
		}
	}

	private void generateVarDecl(Type type, String paramName, SymbolTable symTable, StringBuilder builder) {
		if (type instanceof QualType) {
			String qualifierPrefix = "";
			for (C99Qualifier qualifier : type.get(QualType.C99_QUALIFIERS)) {
				if (qualifier == C99Qualifier.RESTRICT || qualifier == C99Qualifier.RESTRICT_C99) {
					qualifierPrefix = " __restrict__ ";
				}
			}
			generateVarDecl(((QualType) type).getUnqualifiedType(), qualifierPrefix + paramName, symTable, builder);
			if (type.isConst()) {
				builder.append(" const");
			}
			return;
		}
		if (type instanceof PointerType) {
			PointerType pointerType = (PointerType) type;
			generateCodeForType(pointerType.getPointeeType(), symTable, builder);
			if (pointerType.isConst()) {
				builder.append("const");
			}
			builder.append(" *");
			builder.append(paramName);
			return;
		}
		
		generateCodeForType(type, symTable, builder);
		builder.append(" ");
		builder.append(paramName);
	}

	private void generateCodeForType(Type type, SymbolTable symTable, StringBuilder builder) {
		if (type instanceof QualType) {
			QualType qualType = (QualType) type;
			generateCodeForType(qualType.getUnqualifiedType(), symTable, builder);
			if (qualType.isConst()) {
				builder.append(" const");
			}
			for (C99Qualifier qualifier : type.get(QualType.C99_QUALIFIERS)) {
				if (qualifier == C99Qualifier.RESTRICT || qualifier == C99Qualifier.RESTRICT_C99) {
					builder.append(" __restrict__");
				}
			}
			return;
		}
		if (type instanceof PointerType) {
			PointerType pointerType = (PointerType) type;
			generateCodeForType(pointerType.getPointeeType(), symTable, builder);
			builder.append("*");
			return;
		}
		if (type instanceof TypedefType) {
			if (type.getCode().equals("size_t")) {
				builder.append(type.getCode());
				return;
			}
		}
		if (type instanceof ElaboratedType) {
			ElaboratedType elaboratedType = (ElaboratedType) type;
			builder.append("struct ");
			generateCodeForType(elaboratedType.getNamedType(), symTable, builder);
			return;
		}
		if (type instanceof RecordType) {
			RecordType recordType = (RecordType) type;
			builder.append(symTable.getMangledTypeName(recordType.getDecl().getDeclName()));
			return;
		}
		// STUB
		builder.append(type.getCode());
	}
}
