//! # A Density-Based Algorithm for Discovering Clusters
//!
//! This algorithm finds all points within `eps` distance of each other and
//! attempts to cluster them. If there are at least `mpt` points reachable
//! (within distance) from a given point P, then all reachable points are
//! clustered together. The algorithm then attempts to expand the cluster,
//! finding all border points reachable from each point in the cluster
//!
//!
//! See `Ester, Martin, et al. "A density-based algorithm for discovering
//! clusters in large spatial databases with noise." Kdd. Vol. 96. No. 34.
//! 1996.` for the original paper
//!
//! Thanks to the rusty_machine implementation for inspiration

use Classification::{Core, Edge, Noise};

/// Classification according to the DBSCAN algorithm
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Classification {
    /// A point with at least `min_points` neighbors within `max_distance` diameter
    Core(usize),
    /// A point within `max_distance` of a core point, but has less than `min_points` neighbors
    Edge(usize),
    /// A point with no connections
    Noise,
}

/// DBSCAN parameters
pub struct Model<T: ?Sized> {
    range_query: fn(out: &mut Vec<usize>, idx: usize, population: &[&T], eps: f64),
    pub min_cluster_points: usize,
    pub max_distance: f64,
    c: Vec<Classification>,
    n: Vec<usize>,
    v: Vec<bool>,
}

impl<T: ?Sized> Model<T> {
    pub fn new(range_query: fn(out: &mut Vec<usize>, idx: usize, population: &[&T], eps: f64), max_distance: f64, min_cluster_points: usize) -> Model<T> {
        Model { max_distance, min_cluster_points, c: Vec::new(/**/), n: Vec::new(/**/), v: Vec::new(/**/), range_query }
    }

    fn expand_cluster(
        &mut self,
        population: &[&T],
        queue: &mut Vec<usize>,
        cluster: usize,
	    ) -> bool {
	        let mut new_cluster = false;
	        while let Some(ind) = queue.pop() {
                self.n.clear();
	            (self.range_query)(&mut self.n, ind, population, self.max_distance);
	            if self.n.len() < self.min_cluster_points {
	                continue;
	            }
	            new_cluster = true;
                self.c[ind] = Core(cluster);
                for &n_idx in self.n.iter() {
                    // n_idx is at least an edge point
                    if self.c[n_idx] == Noise {
                        self.c[n_idx] = Edge(cluster);
                    }

                    if self.v[n_idx] {
                        continue;
                    }

                    self.v[n_idx] = true;
                    queue.push(n_idx);
                }
            }
        new_cluster
    }

    /// Run the DBSCAN algorithm on a given population of datapoints.
    ///
    /// A vector of [`Classification`] enums is returned, where each element
    /// corresponds to a row in the input matrix.
    ///
    /// # Arguments
    /// * `population` - datapoints organized by row (each row is a slice of coordinates)
    /// ```
    pub fn run(mut self, population: &[&T]) -> Vec<Classification> {
        self.c = vec![Noise; population.len()];
        self.v = vec![false; population.len()];

        let mut cluster = 0;
        let mut queue = Vec::new();

        for idx in 0..population.len() {
            if self.v[idx] {
                continue;
            }

            self.v[idx] = true;

            queue.push(idx);

            if self.expand_cluster(population, &mut queue, cluster) {
                cluster += 1;
            }
        }
        
        self.c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "points must have same dimensionality");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[inline]
    pub fn euclidean_range_query(out: &mut Vec<usize>, idx: usize, population: &[&[f64]], eps: f64) {
        let sample = population[idx];
        let result: Vec<_> = population
            .iter()
            .enumerate()
            .filter(|(_, pt)| euclidean_distance(sample, *pt) < eps)
            .map(|(idx, _)| idx)
            .collect();
        out.extend(result);
    }

    #[test]
    fn cluster() {
        let model = Model::new(euclidean_range_query, 1.0, 3);
        let inputs_vec: Vec<[f64; 2]> = vec![
            [1.5, 2.2],
            [1.0, 1.1],
            [1.2, 1.4],
            [0.8, 1.0],
            [3.7, 4.0],
            [3.9, 3.9],
            [3.6, 4.1],
            [10.0, 10.0],
        ];
        let inputs: Vec<&[f64]> = inputs_vec.iter().map(|pt| pt.as_ref()).collect();
        let output = model.run(&inputs);
        assert_eq!(
            output,
            vec![
                Edge(0),
                Core(0),
                Core(0),
                Core(0),
                Core(1),
                Core(1),
                Core(1),
                Noise
            ]
        );
    }

    #[test]
    fn cluster_edge() {
        let model = Model::new(euclidean_range_query, 0.253110, 3);
        let inputs_vec = vec![
            vec![
                0.3311755015020835,
                0.20474852214361858,
                0.21050489388506638,
                0.23040992344219402,
                0.023161159027037505,
            ],
            vec![
                0.5112445458548497,
                0.1898442816540571,
                0.11674072294944157,
                0.14853288499259437,
                0.03363756454905728,
            ],
            vec![
                0.581134172697341,
                0.15084733646825743,
                0.09997992993087741,
                0.13580335513916678,
                0.03223520576435743,
            ],
            vec![
                0.17210416043100868,
                0.3403172702783598,
                0.18218098373740396,
                0.2616980943829193,
                0.04369949117030829,
            ],
        ];
        let inputs: Vec<&[f64]> = inputs_vec.iter().map(|pt| pt.as_slice()).collect();
        let output = model.run(&inputs);
        assert_eq!(output, vec![Core(0), Core(0), Edge(0), Edge(0)]);
    }
}
