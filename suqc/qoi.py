#!/usr/bin/env python3

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import abc
import os

import pandas as pd
import numpy as np

from suqc.utils.general import cast_series_if_possible
from suqc.configuration import EnvironmentManager
from suqc.resultformat import ParameterResult

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


class QoIProcessor(metaclass=abc.ABCMeta):

    def __init__(self, em: EnvironmentManager, proc_name: str, qoi_name: str):
        self.name = qoi_name

        self._em = em
        self._proc_name = proc_name
        self._proc_config = self._get_proc_config()
        self._proc_id = self._get_ids_proc()
        self._outputfile_name = self._get_outout_filename()


    # TODO: think about time selection option (possibly they can also be located in VADERE directly...)
    def read_and_extract_qoi(self, par_id, output_path) -> ParameterResult:
        data = self._read_csv(output_path)
        data = cast_series_if_possible(data)
        return ParameterResult(par_id, data, self.name)

    def _get_ids_proc(self):
        ids = list()
        for i in self._proc_config:
            ids.append(i["id"])
        return ids

    def _get_all_proc_writers(self):
        return self._em.get_value_basis_file(key="processWriters")[0]

    def _get_proc_config(self):
        procwriter_json = self._get_all_proc_writers()
        procs_list = procwriter_json["processors"]

        return_cfg = list()
        for d in procs_list:
            if d["type"] == self._proc_name:
                return_cfg.append(d)  # append all QoI processors found

                # else:
                #     raise RuntimeError(
                #         "The processor has to be unique to avoid confusion which processor to use for the QoI.")
        if return_cfg is None:
            raise KeyError(f"Could not find QoIProcessor with name {self._proc_name}.")

        return return_cfg

    def _get_outout_filename(self):
        procwriter_json = self._get_all_proc_writers()
        files = procwriter_json["files"]

        file_cfg = None
        for file in files:
            procs_list = file["processors"]

            for pid in self._proc_id:
                if pid in procs_list:
                    # TODO: not best coding... what's needs to be done here:
                    # TODO: There are multiple processor ids and they all need to have defined in the same file (otherwise there is confusion)

                    if file_cfg is None or file_cfg == file:
                        file_cfg = file
                    else:
                        raise RuntimeError("The processor has to have a unique output file, as there are currently "
                                           "only single output files allowed. Multiple files is a feature not "
                                           "implemented yet.")
        return file_cfg["filename"]

    def _filepath(self, output_path):
        return os.path.join(output_path, self._outputfile_name)

    def _read_csv(self, output_path):
        fp = self._filepath(output_path)
        df = pd.read_csv(fp, delimiter=" ", index_col=0, header=0)
        return df


class PedestrianEvacuationTimeProcessor(QoIProcessor):

    def __init__(self, em: EnvironmentManager, apply="mean"):

        assert apply in ["mean", "max"]
        self._apply = apply

        proc_name = "org.vadere.simulator.projects.dataprocessing.processor.PedestrianEvacuationTimeProcessor"
        qoi_name = "_".join(["evacTime", self._apply])

        super(PedestrianEvacuationTimeProcessor, self).__init__(em, proc_name, qoi_name)

    def _apply_homogenization(self, data: pd.Series):
        if self._apply == "mean":
            return data.mean()
        else:
            return data.max()

    def read_and_extract_qoi(self, par_id, output_path):
        data = self._read_csv(output_path)
        data = self._apply_homogenization(data)
        return ParameterResult(par_id, data, self.name)


class PedestrianDensityGaussianProcessor(QoIProcessor):

    def __init__(self, em: EnvironmentManager, apply="mean"):
        proc_name = "org.vadere.simulator.projects.dataprocessing.processor.PedestrianDensityGaussianProcessor"
        assert apply in ["mean", "max"]
        self._apply = apply
        super(PedestrianDensityGaussianProcessor, self).__init__(em, proc_name, "ped_gaussian_density")

    def _apply_homogenization(self, df):
        gb = df.drop("pedestrianId", axis=1).groupby("timeStep")

        if self._apply == "mean":
            return gb.mean()
        else:
            return gb.max()

    def read_and_extract_qoi(self, par_id, output_path):
        df = self._read_csv(output_path)
        df = self._apply_homogenization(df)
        return cast_series_if_possible(df)


class AreaDensityVoronoiProcessor(QoIProcessor):

    def __init__(self, em: EnvironmentManager):
        proc_name = "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityVoronoiProcessor"
        super(AreaDensityVoronoiProcessor, self).__init__(em, proc_name, "voronoiDensity")

    def read_and_extract_qoi(self, par_id, output_path):
        df = self._read_csv(output_path)
        return ParameterResult(par_id, df.T, self.name)


class InitialAndLastPositionProcessor(QoIProcessor):
    """Measures the initial and the last position of an agent."""

    def __init__(self, em: EnvironmentManager):
        proc_name = "org.vadere.simulator.projects.dataprocessing.processor.PedestrianPositionProcessor"
        self._name = "pedPosition"
        super(InitialAndLastPositionProcessor, self).__init__(em, proc_name, self._name)

    def read_and_extract_qoi(self, par_id, output_path):
        df = self._read_csv(output_path)
        assert len(np.unique(df["pedestrianId"])) <= 1, f"For now only single ped. supported, value is " \
                                                        f"{len(np.unique(df['pedestrianId']))} in {output_path}"

        if len(np.unique(df["pedestrianId"])) == 0:
            idx = pd.Index(data=["initial", "last"], name="categ")
            df_first_last = pd.DataFrame(np.nan, index=idx, columns=["x", "y"])
        else:
            df_first_last = df.iloc[[0, -1], :].loc[:, ["x", "y"]]
            df_first_last.index = ["initial", "last"]
            df_first_last.index.name = "categ"

        return ParameterResult(par_id, df_first_last, self._name)


class CountInArea(QoIProcessor):
    # TODO: this has to be actually implemented in VADERE as a Data Processor

    def __init__(self, em, p1, p2):
        proc_name = "org.vadere.simulator.projects.dataprocessing.processor.PedestrianPositionProcessor"
        self._name = "countPed"

        self.p1 = p1
        self.p2 = p2

        super(CountInArea, self).__init__(em, proc_name, self._name)

    def read_and_extract_qoi(self, par_id, output_path):
        data = self._read_csv(output_path)

        x_sec = np.logical_and(data["x"] > self.p1[0], data["x"] < self.p2[0])
        y_sec = np.logical_and(data["y"] > self.p1[1], data["y"] < self.p2[1])
        return ParameterResult(par_id,
                               np.logical_and(x_sec, y_sec).groupby(by="timeStep").apply(lambda x: np.sum(x)),
                               self._name)


class CombinedTwoDensityProcessor(QoIProcessor):
    # TODO: Specialized QoI ... In future the Query should support allow a list of QoIs, but now it is too much!

    def __init__(self, em, p1, p2):
        # No need to call super, because this just combines two QoI -- maybe generalize this!
        self._proc_density = AreaDensityVoronoiProcessor(em)
        self._proc_count = CountInArea(em, p1=p1, p2=p2)
        self._proc_name = "two_density"

    def read_and_extract_qoi(self, par_id, output_path):

        df1 = self._proc_density.read_and_extract_qoi(par_id, output_path).data
        df2 = self._proc_count.read_and_extract_qoi(par_id, output_path).data

        data = np.zeros([3, max(df1.shape[1], df2.shape[1])])

        data[0:2, :df1.shape[1]] = df1.values
        data[2, :df2.shape[1]] = df2.values

        idx = "density1", "density2", "count"
        cols = np.arange(1, max(df1.shape[1], df2.shape[1]) + 1)
        return ParameterResult(par_id, pd.DataFrame(data, index=idx, columns=cols), qoi_name="mixed")

class CustomProcessor(QoIProcessor):

    def __init__(self, em: EnvironmentManager, proc_name: str, qoi_name: str):
        super(CustomProcessor, self).__init__(em, proc_name, qoi_name)


if __name__ == "__main__":
    e = EnvironmentManager("two_density")
    adv = AreaDensityVoronoiProcessor(e)

    pedpos = CountInArea(e)

    print(pedpos)