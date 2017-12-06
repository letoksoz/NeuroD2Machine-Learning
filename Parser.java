import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Parser {
	enum VARIABLE_TYPE {
		NEUROD2, ENCFF145FVU, ENCFF091JOV, ENCFF102IIL, ENCFF676DBG, ENCFF875CQU, ENCFF152TUF, A, T, G, C, TSSupstream, CDS, INTRON, UTR5, UTR3;
	}

	public static void main(String[] args) throws IOException {

		String[] chromType = { "chr1"/*,"chr2"*/};
		for (int i = 0; i < chromType.length; i++) {
			int fileSize = findSize(chromType[i] + ".fa");
			System.out.println("Matrix size: " + fileSize);

			boolean[][] chr = new boolean[VARIABLE_TYPE.values().length][fileSize];

			processHistoneData(chr, chromType[i]);
			System.out.println("processHistoneData completed...");

			processChromeFa(chr, chromType[i] + ".fa");
			System.out.println("processChromeFa completed...");

			processNeuroD2(chr, chromType[i]);
			System.out.println("processNeuroD2 completed...");

			processEnsGeneUniq(chr, chromType[i]);
			System.out.println("processEnsGeneUniq completed...");

			processEnsRegionUniq(chr, chromType[i]);
			System.out.println("processEnsRegionUniq completed...");

			writeDataToFile(chr, chromType[i] + ".data");
			System.out.println("writeToFile completed...");
			
			createSperatedIndex(chr[VARIABLE_TYPE.NEUROD2.ordinal()], chromType[i]);
			
		}

	}

	private static void createSperatedIndex(boolean[] bs, String chromType) throws IOException {
		int numberOfZero = 0;
		for (int i = 0; i < bs.length; i++) {
			if (bs[i] == false) {
				numberOfZero++;
			}
		}
		int numberOfOne = bs.length - numberOfZero;
		int max = Math.max(numberOfOne, numberOfOne);
		
		final int MINI_BATCH_SIZE = 256;
		int numberOfBatchSize = max / (MINI_BATCH_SIZE / 2) + 1;
		
		int indexOfOne = -1;
		int indexOfZero = -1;
		

		// bu indexler julia da bir eklenerek kullanilmali
		int[][] indexes = new int[numberOfBatchSize][MINI_BATCH_SIZE];
		
		for (int i = 0; i < numberOfBatchSize; i++) {
			boolean isZero = false;
			for (int j = 0; j < MINI_BATCH_SIZE; j++) {
				if (isZero) {
					while(true) {
						indexOfZero++;
						indexOfZero %= bs.length;
						if (bs[indexOfZero] == false) {
							indexes[i][j] = indexOfZero;
							break;
						}
					}
				} else {
					while(true) {
						indexOfOne++;
						indexOfOne %= bs.length;
						if (bs[indexOfOne] == true) {
							indexes[i][j] = indexOfOne;
							break;
						}
					}
				}
				isZero = !isZero;
				
			}
					
		}
		writeIndexesToFile(indexes, chromType + "_indexes.txt");
		
	}

	private static void processEnsRegionUniq(boolean[][] chr, String chrType) throws IOException {
		final String ENS_REGION_PATH = "/home/letoksoz/Desktop/JuliaWorkSpace/data/ensRegion_uniq.txt";

		File file = new File(ENS_REGION_PATH);

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while (((line = br.readLine()) != null)) {
			String[] parts = line.split("\t");
			if (parts[0].equals(chrType)) {
				int startIndex = Integer.parseInt(parts[1]);
				int endIndex = Integer.parseInt(parts[2]);
				
				VARIABLE_TYPE type = VARIABLE_TYPE.valueOf(parts[5].toUpperCase());
				
				if (type == null) {
					throw new IllegalArgumentException(parts[5] + " invalid variable type");
				}

				for (int i = startIndex; i <= endIndex; i++) {
					chr[type.ordinal()][i] = true;
				}
			}
		}
		br.close();
	}

	private static void processEnsGeneUniq(boolean[][] chr, String chrType) throws IOException {
		
		final int TSS_UPSTREAM_RANGE = 1000;

		final String ENS_GENE_PATH = "/home/letoksoz/Desktop/JuliaWorkSpace/data/ensGene_uniq.txt";

		File file = new File(ENS_GENE_PATH);

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while (((line = br.readLine()) != null)) {
			String[] parts = line.split("\t");
			if (parts[0].equals(chrType)) {
				int startIndex = Integer.parseInt(parts[1]);
				int endIndex = Integer.parseInt(parts[2]);

				if (parts[5].equals("+")) {
					endIndex = startIndex;
					startIndex = startIndex - TSS_UPSTREAM_RANGE;
				} else if (parts[5].equals("-")) {
					startIndex = endIndex; 
					endIndex = endIndex + TSS_UPSTREAM_RANGE;
				} else {
					throw new IllegalArgumentException(parts[5] + " is not expected.");
				}

				for (int i = startIndex; i < endIndex; i++) {
					chr[VARIABLE_TYPE.TSSupstream.ordinal()][i] = true;
				}
			}
		}
		br.close();
	}

	private static void processNeuroD2(boolean[][] chr, String chrType) throws IOException {
		final String NEURO_DATA_PATH = "/home/letoksoz/Desktop/JuliaWorkSpace/data/neuroD2/";

		String[] neuroD2DataFileName = { "M.txt", "R1.txt", "R2.txt" };

		List<List<Data>> listOfList = new ArrayList<>();

		for (int i = 0; i < neuroD2DataFileName.length; i++) {
			List<Data> list = new ArrayList<>();
			listOfList.add(list);

			File file = new File(NEURO_DATA_PATH + neuroD2DataFileName[i]);

			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			while (((line = br.readLine()) != null)) {
				String[] parts = line.split("\t");
				if (parts[0].equals(chrType)) {
					Data data = new Data(Integer.parseInt(parts[1]), Integer.parseInt(parts[2]));
					list.add(data);
				}
			}
			br.close();
		}

		for (int i = 0; i < listOfList.size(); i++) {
			List<Data> list = listOfList.get(i);

			Collections.sort(list, new Comparator<Data>() {
				@Override
				public int compare(Data d1, Data d2) {
					return d1.start - d2.start;
				}
			});
		}
		List<Data> MFileData = listOfList.get(0);
		List<Data> R1FileData = listOfList.get(1);
		List<Data> R2FileData = listOfList.get(2);
		
		int maxIndex = Math.max(Math.max(MFileData.get(MFileData.size() - 1).end, R1FileData.get(R1FileData.size() - 1).end),  
				R2FileData.get(R2FileData.size() - 1).end);
		System.out.println("Max index of NEUROD2 file: " + maxIndex + " for "+ chrType);
		
		boolean[] MFileBoolArray = convertToBoolArray(MFileData, maxIndex);
		boolean[] R1FileBoolArray = convertToBoolArray(R1FileData, maxIndex);
		boolean[] R2FileBoolArray = convertToBoolArray(R1FileData, maxIndex);
		
		boolean[] intersectionBoolArray = findIntersectionList(MFileBoolArray, R1FileBoolArray, R2FileBoolArray, maxIndex);
		
		for (int i = 0; i < maxIndex; i++) {
			chr[VARIABLE_TYPE.NEUROD2.ordinal()][i] = intersectionBoolArray[i];
		}
	}
	
	private static boolean[]  findIntersectionList(boolean[]  MFileBoolArray, boolean[] R1FileBoolArray, boolean[] R2FileBoolArray, int maxIndex) {
		
		boolean[] listArray = new boolean[maxIndex];
		for (int i = 0; i < maxIndex; i++) {
			listArray[i] = MFileBoolArray[i] && R1FileBoolArray[i] & R2FileBoolArray[i];
		}
		return listArray;
	}

	private static boolean[] convertToBoolArray(List<Data> list1, int maxIndex) {
		boolean[] list1Array = new boolean[maxIndex];
		for (Data data : list1) {
			for (int i = data.start; i < data.end; i++) {
				list1Array[i] = true;
			}
		}
		return list1Array;
	}

	private static int findSize(String fileName) throws IOException {
		final String CHROM_FA_DATA_PATH = "/home/letoksoz/Desktop/JuliaWorkSpace/data/chromFa/";

		BufferedReader br = new BufferedReader(new FileReader(CHROM_FA_DATA_PATH + fileName));
		String line;
		int characterCounter = 0;
		
		
		while (((line = br.readLine()) != null)) {
			if (line.startsWith(">")) {
				// skip the first line
				continue;
			}
			characterCounter += line.length();
		}
		br.close();
		return characterCounter;
	}
	private static void processChromeFa(boolean[][] chr, String fileName) throws IOException {
		final String CHROM_FA_DATA_PATH = "/home/letoksoz/Desktop/JuliaWorkSpace/data/chromFa/";

		BufferedReader br = new BufferedReader(new FileReader(CHROM_FA_DATA_PATH + fileName));
		String line;
		int characterCounter = 0;
		
		while (((line = br.readLine()) != null)) {
			if (line.startsWith(">")) {
				// skip the first line
				continue;
			}
			for (int i = 0; i < line.length(); i++) {
				char ch = Character.toUpperCase(line.charAt(i));

				switch (ch) {
				case 'A':
					chr[VARIABLE_TYPE.A.ordinal()][characterCounter] = true;
					break;
				case 'T':
					chr[VARIABLE_TYPE.T.ordinal()][characterCounter] = true;
					break;
				case 'G':
					chr[VARIABLE_TYPE.G.ordinal()][characterCounter] = true;
					break;
				case 'C':
					chr[VARIABLE_TYPE.C.ordinal()][characterCounter] = true;
					break;
				case 'N':
					// Do nothing
					break;
				default:
					throw new IllegalArgumentException(ch + " is not a chrom type");
				}
				characterCounter++;
			}
		}
		br.close();

		System.out.println("There are " + characterCounter + " characters in the file ");
	}

	private static void processHistoneData(boolean[][] chr, String chrType) throws FileNotFoundException, IOException {
		long startTotalTime = System.currentTimeMillis();
		int totalLineCounter = 0;

		final String HISTONE_DATA_PATH = "/home/letoksoz/Desktop/JuliaWorkSpace/data/histone_data/";
		final double HISTONE_THRESHOLD = 0;
		final int NUMBER_OF_LINE_TO_LOG = 30_000_000;

		String[] histoneDataFileName = { "ENCFF145FVU.bedgraph", "ENCFF091JOV.bedgraph", "ENCFF102IIL.bedgraph",
				"ENCFF676DBG.bedgraph", "ENCFF875CQU.bedgraph", "ENCFF152TUF.bedgraph" };

		for (int i = 0; i < histoneDataFileName.length; i++) {
			long startTime = System.currentTimeMillis();
			int lineCounter = 0;
			File file = new File(HISTONE_DATA_PATH + histoneDataFileName[i]);

			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			while (((line = br.readLine()) != null)) {
				lineCounter++;
				if (lineCounter % NUMBER_OF_LINE_TO_LOG == 0) {
					System.out.printf("Processed number of line: %12d takes %6d seconds\n", lineCounter,
							(System.currentTimeMillis() - startTime) / 1000);
				}
				String[] parts = line.split("\t");
				if (parts[0].equals(chrType)) {
					int start = Integer.parseInt(parts[1]);
					int end = Integer.parseInt(parts[2]);
					double dValue = Double.parseDouble(parts[3]);

					// TODO histogram data analizi yapilip ona gore bu karar verilecek
					boolean value = (dValue != HISTONE_THRESHOLD);

					for (int j = start; j < end; j++) {
						chr[i][j] = value;
					}
				}
			}
			br.close();

			totalLineCounter += lineCounter;

			System.out.println("File name: " + histoneDataFileName[i]);
			System.out.println("Number of lines: " + lineCounter);
			System.out.println("Time in second: " + (System.currentTimeMillis() - startTime) / 1000);
		}

		System.out.println("Total number of lines: " + totalLineCounter);
		System.out.println("Total time in second: " + (System.currentTimeMillis() - startTotalTime) / 1000);
	}

	private static void writeDataToFile(boolean[][] chr, String fileName) throws IOException {
		int counter = 0;
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		for (int i = 0; i < chr.length; i++) {
			for (int j = 0; j <= chr[0].length; j++) {
				if (chr[i][j]) {
					writer.append('1');
					counter++;
				} else {
					writer.append('0');
				}
			}
			writer.append('\n');
		}
		writer.close();
		System.out.println("total number of trues: " + counter);
	}
	
	private static void writeIndexesToFile(int[][] indexes, String fileName) throws IOException {
		final double TRAINING_RATIO = 0.90; 
		
		int trainingSize = (int)(indexes.length * TRAINING_RATIO);
		
		
		BufferedWriter writer = new BufferedWriter(new FileWriter("training_" + fileName));
		
		for (int i = 0; i < trainingSize; i++) {
			for (int j = 0; j < indexes[i].length; j++) {
				writer.append(indexes[i][j] + " "); // 23 56 24 58 25 59 
			}
			writer.append("\n");
		}
		writer.close();

		
		writer = new BufferedWriter(new FileWriter("test_" + fileName));
		
		for (int i = trainingSize; i < indexes.length; i++) {
			for (int j = 0; j < indexes[i].length; j++) {
				writer.append(indexes[i][j] + " "); 
			}
			writer.append("\n");
		}
		writer.close();
	}
}

class Data {
	int start;
	int end;

	public Data(int start, int end) {
		if (end < start) {
			int temp = start;
			start = end;
			end = temp;
		}
		this.start = start;
		this.end = end;
	}
}
