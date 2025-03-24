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

        String dataset_name = "sumo_mixed_car_vs_bikes";
        System.out.println(dataset_name);

        //args attesi: dataset, #thread, #description file
        long start = System.currentTimeMillis();
        String descriptionFile_path = "spatialMovelets.json";
        String trajectoriesPath_train = "../dataset_for_movelets/"+dataset_name+"/train_"+dataset_name;
        String trajectoriesPath_test = "../dataset_for_movelets/"+dataset_name+"/test_"+dataset_name;

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
        int nthreads = 128;

        MoveletsMultithread analysis = new
                MoveletsMultithread(
                traiettorie_train, traiettorie_test, dms, minSize,
                100,
                nthreads,
                qualityMeasure, useCache, "../dataset_for_movelets/"+dataset_name+"/res_"+dataset_name);

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
