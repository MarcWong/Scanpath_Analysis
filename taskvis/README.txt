===============
TASKVIS dataset
===============

This dataset corresponds to the task-based visual analysis experiment described in the following paper:

```
Polatsek, P., Waldner, M., Viola, I., Kapec, P., & Benesova, W. (2018)
Exploring Visual Attention and Saliency Modeling for Task-Based Visual Analysis
Computers & Graphics
doi:10.1016/j.cag.2018.01.010
```

Please cite this paper if you use this dataset.

This dataset contains eye-tracking data from 47 participants who solved three low-level analytic tasks with 30 selected visualizations from MASSVIS dataset (the visualizations and eye-tracking data of the original memorability experiment can be found at http://massvis.mit.edu/ ).

---------------------
Experiment procedure:
---------------------
Participants were shown 30 different visualizations. For each visualization, they solved one of three task types. First, participants were shown a task description. Then, they saw a visualization and finally, they entered their answer.

----------------------------
Contents of TASKVIS dataset:
----------------------------
- images.txt		visualizations used in the experiment
- tasks.txt			task descriptions
- tasktypes.txt		task types solved by participants
- answers.txt		correctness of participant's answers
- fixations/		eye-tracking data
- labels/			task-dependent AOIs

----------
images.txt
----------
This file contains the list of all visualizations used in the experiment with their corresponding IDs. This file contains the following columns:
- imageID			visualization ID
- filename			visualization name

---------
tasks.txt
---------
This file contains descriptions of all tasks used in the experiment. There are three task types for each visualization: retrieve-value-task (RV), filter-task (F) and find-extremum-task (FE). This file contains the following columns:
- imageID			visualization ID
- task				task type (A = RV-task, B = F-task, C = FE-task)
- question			task description

-------------
tasktypes.txt
-------------
This file contains the list of tasks solved by each participant per line (A = RV-task, B = F-task, C = FE-task). The file contains the following columns (sorted by visualization ID):
- user				user ID
- 1					task type for visualization #1
- 2					task type for visualization #2
...
- 30				task type for visualization #30

-----------
answers.txt
-----------
This file contains answers of each participant per line (1 = correct, 0 = incorrect). The file contains the following columns (sorted by visualization ID):
- user				user ID
- 1					correctness of the answer for visualization #1 
- 2					correctness of the answer for visualization #2 
...
- 30				correctness of the answer for visualization #30 

----------
fixations/
----------
This folder contains participants' fixations when performing a task for a visualization. Each file consists of fixations of a particular participant when viewing a particular visualization.

File format: rec_user_fix_imageID.tsv:
- user				user ID
- imageID			visualization ID

e.g. rec_p02_fix_3.tsv - fixations of user P02 for visualization #3

Each file contains the following columns:
- RecordingTimestamp		timestamp in msec counted from the start of the image display
- FixationIndex				order in which fixation was recorded inside the image (starting with 1)
- FixationPointX (MCSpx)	horizontal image coordinate of the fixation point in pixels (origin in the top left corner of the image)
- FixationPointY (MCSpx)	vertical image coordinate of the fixation point in pixels (origin in the top left corner of the image)

The last row contains the duration of image display in msec. 

-------
labels/
-------
This folder contains positions of task-dependent AOIs for each visualization per file. The AOIs need to be attended to correctly answer the question. Each file contains AOI names and image coordinates of vertices per lines.

Format of AOI names: taskType_AOItype
- taskType			a = RV-task, b = F-task, c = FE-task
- AOItype			value label, value annotation, value legend, data point, item label, item legend 
							