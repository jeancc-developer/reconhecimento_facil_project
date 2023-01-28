package reconhecimento;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class Treinamento {
	public static void main(String[] args) {
		File diretorio = new File("src\\fotos");
		FilenameFilter filtroImagem = new FilenameFilter() {

			@Override
			public boolean accept(File dir, String nome) {
				return nome.endsWith(".jpg") || nome.endsWith(".gif") || nome.endsWith(".png");
			}
		};

		File[] arquivos = diretorio.listFiles(filtroImagem);
		MatVector fotos = new MatVector(arquivos.length);
		Mat rotulos = new Mat(arquivos.length, 1, CV_32SC1);
		IntBuffer rotulosBuffer = rotulos.createBuffer();
		int contador = 0;
		for (File imagem : arquivos) {
			Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);
			int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
			// System.out.println(classe);
			resize(foto, foto, new Size(160, 160));
			fotos.put(contador, foto);
			rotulosBuffer.put(contador, classe);
			contador++;
		}

		FaceRecognizer eigenFaces = EigenFaceRecognizer.create(10, 0);
		FaceRecognizer fisherFaces = FisherFaceRecognizer.create();
		FaceRecognizer lbph = LBPHFaceRecognizer.create(2,9,9,9,1);

		eigenFaces.train(fotos, rotulos);
		eigenFaces.save("src\\recursos\\classificarEigenFaces.yml");
		fisherFaces.train(fotos, rotulos);
		fisherFaces.save("src\\recursos\\classificadorFisherFaces.yml");
		lbph.train(fotos, rotulos);
		lbph.save("src\\recursos\\classificadorLBPH.yml");

	}

}
