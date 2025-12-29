#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <gng_server.h>
#include <gng_configuration.h>
#include <gng_algorithm.h>
#include <gng_graph.h>
#include <gng_node.h>
#include <utils/utils.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using Array2d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Array1d = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

class GraphLockGuard {
public:
	explicit GraphLockGuard(GNGGraph &graph) : graph_(graph) {
		graph_.lock();
	}

	~GraphLockGuard() {
		graph_.unlock();
	}

private:
	GNGGraph &graph_;
};

template <typename Array>
std::vector<double> copy_flat_array(const Array &array, const std::string &name,
		size_t expected_size) {
	auto buf = array.request();
	if (buf.ndim != 1) {
		throw py::value_error(name + " must be a 1-D array");
	}
	if (static_cast<size_t>(buf.shape[0]) != expected_size) {
		throw py::value_error("Expected " + std::to_string(expected_size) + " values in " + name);
	}
	std::vector<double> result(expected_size);
	std::memcpy(result.data(), buf.ptr, expected_size * sizeof(double));
	return result;
}

py::dict build_node_dict(GNGNode &node) {
	py::dict data;
	data["index"] = node.nr;
	data["error"] = node.error;
	data["label"] = node.extra_data;
	data["utility"] = node.utility;
	data["position"] = std::vector<double>(node.position, node.position + node.dim);
	std::vector<int> neighbours;
	neighbours.reserve(node.size());
	for (GNGEdge *edge : node) {
		neighbours.push_back(edge->nr);
	}
	data["neighbours"] = neighbours;
	return data;
}

py::dict get_node_snapshot(GNGServer &server, int index) {
	if (index < 0) {
		throw py::value_error("node index must be non-negative");
	}

	GNGGraph &graph = server.getGraph();
	GraphLockGuard guard(graph);
	if (!graph.existsNode(static_cast<unsigned int>(index))) {
		throw py::value_error("node index does not exist");
	}
	return build_node_dict(graph[index]);
}

py::list collect_nodes(GNGServer &server) {
	GNGGraph &graph = server.getGraph();
	GraphLockGuard guard(graph);
	py::list nodes;
	const unsigned int max_index = graph.get_maximum_index();
	for (unsigned int i = 0; i <= max_index; ++i) {
		if (!graph.existsNode(i)) {
			continue;
		}
		nodes.append(build_node_dict(graph[i]));
	}
	return nodes;
}

void insert_examples(GNGServer &server, Array2d data, py::object labels,
		py::object probabilities) {
	auto buf = data.request();
	if (buf.ndim != 2) {
		throw py::value_error("data must be a 2-D array");
	}
	const size_t count = static_cast<size_t>(buf.shape[0]);
	const size_t dim = static_cast<size_t>(buf.shape[1]);

	const auto config = server.getConfiguration();
	if (dim != static_cast<size_t>(config.dim)) {
		throw py::value_error(
				"data dimensionality does not match configuration (expected " + std::to_string(config.dim)
						+ ")");
	}

	std::vector<double> positions(count * dim);
	std::memcpy(positions.data(), buf.ptr, positions.size() * sizeof(double));

	double *labels_ptr = nullptr;
	std::vector<double> labels_storage;
	if (!labels.is_none()) {
		Array1d lbl = labels.cast<Array1d>();
		labels_storage = copy_flat_array(lbl, "labels", count);
		labels_ptr = labels_storage.data();
	}

	double *prob_ptr = nullptr;
	std::vector<double> prob_storage;
	if (!probabilities.is_none()) {
		Array1d prob = probabilities.cast<Array1d>();
		prob_storage = copy_flat_array(prob, "probabilities", count);
		prob_ptr = prob_storage.data();
	}

	{
		py::gil_scoped_release release;
		server.insertExamples(positions.data(), labels_ptr, prob_ptr,
				static_cast<unsigned int>(count), static_cast<unsigned int>(dim));
	}
}

int predict_sample(GNGServer &server, Array1d sample) {
	auto buf = sample.request();
	if (buf.ndim != 1) {
		throw py::value_error("sample must be a 1-D array");
	}
	const size_t dim = static_cast<size_t>(buf.shape[0]);
	const auto config = server.getConfiguration();
	if (dim != static_cast<size_t>(config.dim)) {
		throw py::value_error(
				"sample dimensionality does not match configuration (expected " + std::to_string(config.dim)
						+ ")");
	}

	std::vector<double> values(dim);
	std::memcpy(values.data(), buf.ptr, dim * sizeof(double));
	return server.getAlgorithm().predict(values);
}

void update_clustering(GNGServer &server) {
	py::gil_scoped_release release;
	server.getAlgorithm().updateClustering();
}

std::vector<int> get_clustering(GNGServer &server) {
	const std::vector<int> &assignments = server.getAlgorithm().get_clustering();
	return assignments;
}

} // namespace

PYBIND11_MODULE(_core, m) {
	m.doc() = "Pybind11 bindings for Growing Neural Gas";

	py::register_exception<BasicException>(m, "GNGError");

	py::class_<GNGConfiguration> config_cls(m, "GNGConfiguration");
	config_cls
			.def(py::init<>())
			.def_readwrite("max_nodes", &GNGConfiguration::max_nodes)
			.def_readwrite("uniformgrid_optimization", &GNGConfiguration::uniformgrid_optimization)
			.def_readwrite("lazyheap_optimization", &GNGConfiguration::lazyheap_optimization)
			.def_readwrite("dim", &GNGConfiguration::dim)
			.def_readwrite("orig", &GNGConfiguration::orig)
			.def_readwrite("axis", &GNGConfiguration::axis)
			.def_readwrite("max_age", &GNGConfiguration::max_age)
			.def_readwrite("alpha", &GNGConfiguration::alpha)
			.def_readwrite("beta", &GNGConfiguration::beta)
			.def_readwrite("lambda_", &GNGConfiguration::lambda)
			.def_readwrite("grow_on_new_samples", &GNGConfiguration::grow_on_new_samples)
			.def_readwrite("new_node_position_mode", &GNGConfiguration::new_node_position_mode)
			.def_readwrite("eps_w", &GNGConfiguration::eps_w)
			.def_readwrite("eps_n", &GNGConfiguration::eps_n)
			.def_readwrite("verbosity", &GNGConfiguration::verbosity)
			.def_readwrite("distance_function", &GNGConfiguration::distance_function)
			.def_readwrite("dataset_type", &GNGConfiguration::datasetType)
			.def_readwrite("starting_nodes", &GNGConfiguration::starting_nodes)
			.def_readwrite("experimental_utility_k", &GNGConfiguration::experimental_utility_k)
			.def_readwrite("experimental_utility_option", &GNGConfiguration::experimental_utility_option)
			.def("set_bounding_box", &GNGConfiguration::setBoundingBox, py::arg("min_val"),
					py::arg("max_val"))
			.def("check", &GNGConfiguration::check_correctness,
					"Check whether the configuration is internally consistent.");

	py::class_<GNGServer>(m, "GNGServer")
			.def(py::init([](const GNGConfiguration &config) {
				return std::unique_ptr<GNGServer>(new GNGServer(config, nullptr));
			}), py::arg("config"))
			.def(py::init<const std::string &>(), py::arg("filename"))
			.def("run", [](GNGServer &self) {
				py::gil_scoped_release release;
				self.run();
			})
			.def("pause", [](GNGServer &self) {
				py::gil_scoped_release release;
				self.pause();
			})
			.def("terminate", [](GNGServer &self) {
				py::gil_scoped_release release;
				self.terminate();
			})
			.def("is_running", &GNGServer::isRunning)
			.def("has_started", &GNGServer::hasStarted)
			.def("current_iteration", &GNGServer::getCurrentIteration)
			.def("dataset_size", &GNGServer::getDatasetSize)
			.def("number_of_nodes", &GNGServer::getNumberNodes)
			.def("mean_error", &GNGServer::getMeanError)
			.def("mean_error_statistics", &GNGServer::getMeanErrorStatistics)
			.def("error_index", &GNGServer::getGNGErrorIndex)
			.def("configuration", &GNGServer::getConfiguration)
			.def("save", [](GNGServer &self, const std::string &path) {
				py::gil_scoped_release release;
				self.save(path);
			}, py::arg("path"))
			.def("export_graphml", [](GNGServer &self, const std::string &path) {
				py::gil_scoped_release release;
				self.exportToGraphML(path);
			}, py::arg("path"))
			.def("node", &get_node_snapshot, py::arg("index"))
			.def("nodes", &collect_nodes)
			.def("insert_examples", &insert_examples, py::arg("data"), py::arg("labels") = py::none(),
					py::arg("probabilities") = py::none())
			.def("predict", &predict_sample, py::arg("sample"))
			.def("update_clustering", &update_clustering)
			.def("clustering", &get_clustering)
			.def("graph_distance", [](GNGServer &self, int node_a, int node_b) {
				if (node_a < 0 || node_b < 0) {
					throw py::value_error("node indices must be non-negative");
				}
				return self.nodeDistance(node_a + 1, node_b + 1);
			}, py::arg("node_a"), py::arg("node_b"));

	m.def("load_gng", [](const std::string &filename) {
		return std::unique_ptr<GNGServer>(new GNGServer(filename));
	}, py::arg("filename"), "Load a serialized GNGServer from disk.");
}
