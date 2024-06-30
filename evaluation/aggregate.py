import logging
import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path


#
# go through each .json file in the first folder
# find the files with the same name in the other folders
#
# Build one data structure with all the values for each of the experiments
# calculate average and std.dev
#
# store the aggregated values as .json files in the root of experiments
#

label_tests = ["test.json", "hs5.json", "hs6.json", "os.json", "ou.json"]


def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("path", help="path to experiment folder")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print("Cannot find experiments folder: {}. Exiting ...".format(path))
        exit()

    path0 = path / "0"

    # find the experiment file names
    # the names shall be equal for every folder
    file_names = [f.name for f in path0.iterdir() if f.suffix == ".json"]
    logging.info("file_names: {}".format(file_names))
    dirs = [d for d in path.iterdir() if d.is_dir()]
    dirs.sort()  # keep the order of the folders stable
    logging.info("dirs: {}".format(dirs))

    num_experiments = len(dirs)

    # for each experiment
    # collect the data from each fold
    for f in file_names:
        results = {}

        # populate the results dict with all the results from all experiments
        for d in dirs:
            current_path = d / f
            logging.info("Reading results from %s", current_path)
            with current_path.open() as F:
                data = json.load(F)
                logging.info("Read results from %s", current_path)

            if f in label_tests:  # the normal test (not scd elements)
                for label in data.keys():
                    if not label in results:
                        results[label] = {}
                    for metric, value in data[label].items():
                        if metric not in results[label]:
                            results[label][metric] = []
                        results[label][metric].append(value)

            else:
                #print("data: {}".format(data))
                for key, value in data.items():
                    if not key in results:
                        results[key] = []
                    results[key].append(value)

        #print("results: {}".format(results))
        # create new dict with aggregates and std.dev
        if f in label_tests:  # the normal test (not scd elements)
            agg_res = {}
            for label in results.keys():
                if not label in agg_res:
                    agg_res[label] = {}
                for metric, res in results[label].items():
                    if not metric in agg_res[label]:
                        agg_res[label][metric] = {}
                    res_arr = np.array(res)
                    agg_res[label][metric]['mean'] = res_arr.mean()
                    agg_res[label][metric]['stdev'] = res_arr.std()
                    agg_res[label][metric]['results'] = res_arr.tolist()
        else:
            agg_res = {}
            for key, results in results.items():
                if not key in agg_res:
                    agg_res[key] = {}
                res_arr = np.array(results)
                #print("res_arr: {}".format(res_arr))
                agg_res[key]['mean'] = res_arr.mean()
                agg_res[key]['stdev'] = res_arr.std()
                agg_res[key]['results'] = res_arr.tolist()

        # write file with aggregated result to root folder
        agg_path = path / "agg_{}".format(f)
        with agg_path.open(mode='w') as F:
            json.dump(agg_res, F, indent=4)
            logging.info("Written aggregated results to %s", agg_path)

    logging.info("Finished aggregating all results")






if __name__ == '__main__':
    main()
