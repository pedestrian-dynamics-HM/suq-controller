import os


class RequestItem(object):
    def __init__(self, parameter_id, run_id, scenario_path, base_path, output_folder):
        self.parameter_id = parameter_id
        self.run_id = run_id
        self.base_path = base_path
        self.output_folder = output_folder
        self.scenario_path = scenario_path

        self.output_path = os.path.join(self.base_path, self.output_folder)

    def add_qoi_result(self, qoi_result):
        self.qoi_result = qoi_result

    def add_meta_info(self, required_time, return_code):
        self.required_time = required_time
        self.return_code = return_code
