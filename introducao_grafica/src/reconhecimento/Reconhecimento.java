package reconhecimento;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_PLAIN;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;

import java.awt.event.KeyEvent;
import java.util.Scanner;

import org.bytedeco.opencv.opencv_face.*;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Reconhecimento {

	public static void main(String[] args) throws Exception, InterruptedException {
		//KeyEvent tecla = null;
		OpenCVFrameConverter.ToMat converteToMat = new OpenCVFrameConverter.ToMat();
		OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
		String[] pessoas = {"", "Jean", "Carlos"};  
		camera.start();

		CascadeClassifier detectarFace = new CascadeClassifier("src\\recursos\\haarcascade_frontalface_alt.xml");
		
		//FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
		//reconhecedor.read("src\\recursos\\classificarEigenFaces.yml");
		
//		FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
//		reconhecedor.read("src\\recursos\\classificadorFisherFaces.yml");
		
		FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
		reconhecedor.read("src\\recursos\\classificadorLBPH.yml");

		CanvasFrame cFrame = new CanvasFrame("Reconhecimento", CanvasFrame.getDefaultGamma() / camera.getGamma());
		Frame frameCapturado = null;
		Mat imagemColorida = new Mat();
		/*
		 * int numeroAmostras = 25; int amostra = 1;
		 * System.out.println("Digite o seu id: "); Scanner cadastro = new
		 * Scanner(System.in); int idPessoa = cadastro.nextInt();
		 */
		while ((frameCapturado = camera.grab()) != null) {
			imagemColorida = converteToMat.convert(frameCapturado);
			Mat imagemCinza = new Mat();
			cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
			RectVector facesDetectadas = new RectVector();
			detectarFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150),
					new Size(500, 500));
//			if (tecla == null) {
//				tecla = cFrame.waitKey(5);
//			}
			for (int i = 0; i < facesDetectadas.size(); i++) {
				Rect dadosFace = facesDetectadas.get(0);
				rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));
				Mat faceCapturada = new Mat(imagemCinza, dadosFace);
				resize(faceCapturada, faceCapturada, new Size(160, 160));

				IntPointer rotulos = new IntPointer(1);
				DoublePointer confianca = new DoublePointer(1);
				reconhecedor.predict(faceCapturada, rotulos, confianca);
				int predicao = rotulos.get(0);
				String nome;
				if (predicao == -1) {
					nome = "Desconhecido";
				} else {
					nome = pessoas[predicao] + " - " + confianca.get(0);
				}
				
				int x = Math.max(dadosFace.tl().x() - 10, 0);
				int y = Math.max(dadosFace.tl().y() - 10, 0);
				putText(imagemColorida, nome, new Point(x,y), FONT_HERSHEY_PLAIN, 1.4, new Scalar(0,255,0,0));

			}
			if (cFrame.isVisible()) {
				cFrame.showImage(frameCapturado);
			}

			/*
			 * if (amostra > numeroAmostras) { break; }
			 */
		}
		cFrame.dispose();
		camera.stop();
	}

}
