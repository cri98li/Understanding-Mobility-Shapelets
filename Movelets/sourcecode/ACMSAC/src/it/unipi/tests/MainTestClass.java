package it.unipi.tests;

import br.ufsc.trajectoryclassification.model.bo.dmbs.DMS;
import br.ufsc.trajectoryclassification.model.bo.dmbs.IDistanceMeasureForSubtrajectory;
import br.ufsc.trajectoryclassification.model.bo.movelets.MoveletsMultithread;
import br.ufsc.trajectoryclassification.model.bo.movelets.MyCounter;
import br.ufsc.trajectoryclassification.model.bo.movelets.QualityMeasures.IQualityMeasure;
import br.ufsc.trajectoryclassification.model.bo.movelets.QualityMeasures.InformationGain;
import br.ufsc.trajectoryclassification.model.dao.TrajectoryDAO;
import br.ufsc.trajectoryclassification.model.vo.ITrajectory;
import br.ufsc.trajectoryclassification.model.vo.description.Description;
import br.ufsc.trajectoryclassification.utils.Utils;
import weka.core.pmml.jaxbbindings.False;

import java.util.List;

public class MainTestClass {
    public static void main(String[] args){
        //args attesi: dataset, #thread, #description file
        long start = System.currentTimeMillis();
        String descriptionFile_path = "spatialMovelets.json";
        String trajectoriesPath_train = "train";
        String trajectoriesPath_test = "test";

        Description description = new TrajectoryDAO().loadDescription(descriptionFile_path);
        List<ITrajectory> traiettorie_train = Utils.loadTrajectories(trajectoriesPath_train, description);

        if (traiettorie_train.isEmpty()) {
            System.out.println("Empty training set");
            return;
        }

        List<ITrajectory> traiettorie_test = Utils.loadTrajectories(trajectoriesPath_test, description);

        if (traiettorie_test.isEmpty()) {
            System.out.println("Empty test set");
            return;
        }

        IDistanceMeasureForSubtrajectory dms =  DMS.getDMSfromDescription(description);


        boolean useCache = false;
        IQualityMeasure qualityMeasure = new InformationGain(traiettorie_train); //new LeftSidePure(traiettorie_train);
        int minSize = 2;
        int maxSize = 13;
        int nthreads = 20;

        MoveletsMultithread analysis = new
                MoveletsMultithread(
                traiettorie_train, traiettorie_test, dms, minSize, maxSize, nthreads, qualityMeasure, useCache, "results/r/");

        analysis.run();

        MyCounter.data.put("candidates", MyCounter.numberOfCandidates);

        System.out.println(MyCounter.data);

        System.out.println("\r\n\r\n\r\nTIME: "+ (System.currentTimeMillis() - start) + "ms");
    }
}

/*
-------------------------- Esempio description file

{
   "readsDesc": [
        {
          "order": 1,
          "type": "numeric",
          "text": "time"
        },
        {
          "order": 2,
          "type": "space2d",
          "text": "space"
        }
      ]
}

 */

/*
Buonasera Professore,
ho provato ad impostare il problema come mi ha detto. Nello specifico, dopo aver pulito il dataset su Pechino, per ogni dataset ho:
-eliminato gli id delle traiettorie che contenevano dei "valori strani" in quanto sospetti outliers
-lat/long: "centrato" le traiettorie sottraendo la media
-data: ho preso il giorno con il timestamp minimo, calcolato la mezzanotte di quel giorno ed ho usato il risultato come data zero (dato che il minimo potrebbe essere un outlier, ho fatto anche un secondo dataset dove ho usato la media, ma il risultato non cambia)

Ho poi unito i datasets, estratto le movelets e usato un decision tree per la classificazione e, anche offrendo pochissimi dati per l'estrazione delle movelets, la classificazione risulta quasi sempre con accuracy maggiore del 80% (dati bilanciati).


Potrebbe essere un problema per quanto riguarda le trasformazioni che ho applicato? è il problema di classificazione che è semplice?
 */