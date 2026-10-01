use crate::{
    dispatch::backend::{
        NodeId,
        kernel::{Dependencies, Redirect},
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};
use std::{collections::HashMap, vec::Vec};

/// Sorts a graph-like structure from these dependencies, assuming `dep` is the index in this slice
pub fn topo_sort<T: Clone>(
    nodes: &[Dependencies<T>],
) -> Result<Vec<Dependencies<T>>, Error<GraphErrorContext<'_>>> {
    let n = nodes.len();

    let mut in_degree = vec![0_usize; n];
    let mut adj = vec![Vec::new(); n];

    for (node_id, node) in nodes.iter().enumerate() {
        for &inp in &node.dep {
            if inp >= n {
                return Err(Error {
                    msg: "invalid node reference in graph",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::MissingInput {
                        node: node_id,
                        input: inp,
                    },
                });
            }

            adj[inp].push(node_id);
            in_degree[node_id] += 1;
        }
    }

    let mut zeros: Vec<_> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    zeros.sort_unstable();

    let mut order = Vec::with_capacity(n);

    let mut idx = 0;
    while idx < zeros.len() {
        let node = zeros[idx];
        idx += 1;

        order.push(node);

        let mut nexts = adj[node].clone();
        nexts.sort_unstable();

        for nxt in nexts {
            in_degree[nxt] -= 1;
            if in_degree[nxt] == 0 {
                zeros.push(nxt);
            }
        }
    }

    if order.len() != n {
        return Err(Error {
            msg: "cycle detected in graph",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::CycleDetected {
                node: 0,
                path: order,
            },
        });
    }

    let mut new_index = vec![0_usize; n];
    for (i, &old) in order.iter().enumerate() {
        new_index[old] = i;
    }

    let mut new_nodes = Vec::with_capacity(n);

    for &old_id in &order {
        let mut node = nodes[old_id].clone();

        for inp in &mut node.dep {
            *inp = new_index[*inp];
        }

        new_nodes.push(node);
    }

    Ok(new_nodes)
}

/// Sorts a graph just like [`topo_sort`], but updating redirections as well
///
/// Assumes redirections and dependencies are indices into the graph
pub fn topo_sort_indirect<T: Clone>(
    nodes: &[Dependencies<Redirect<T>>],
) -> Result<Vec<Dependencies<Redirect<T>>>, Error<GraphErrorContext<'_>>> {
    let n = nodes.len();

    let mut in_degree = vec![0_usize; n];
    let mut adj = vec![Vec::new(); n];

    for (node_id, node) in nodes.iter().enumerate() {
        for &inp in &node.dep {
            if inp >= n {
                return Err(Error {
                    msg: "invalid node reference in graph",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::MissingInput {
                        node: node_id,
                        input: inp,
                    },
                });
            }

            adj[inp].push(node_id);
            in_degree[node_id] += 1;
        }
    }

    let mut zeros: Vec<_> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    zeros.sort_unstable();

    let mut order = Vec::with_capacity(n);

    let mut idx = 0;
    while idx < zeros.len() {
        let node = zeros[idx];
        idx += 1;

        order.push(node);

        let mut nexts = adj[node].clone();
        nexts.sort_unstable();

        for nxt in nexts {
            in_degree[nxt] -= 1;

            if in_degree[nxt] == 0 {
                zeros.push(nxt);
            }
        }
    }

    if order.len() != n {
        return Err(Error {
            msg: "cycle detected in graph",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::CycleDetected {
                node: 0,
                path: order,
            },
        });
    }

    let mut new_index = vec![0_usize; n];

    for (new, &old) in order.iter().enumerate() {
        new_index[old] = new;
    }

    let mut new_nodes = Vec::with_capacity(n);

    for &old_id in &order {
        let mut node = nodes[old_id].clone();

        for inp in &mut node.dep {
            *inp = new_index[*inp];
        }

        if let Redirect::Redirected(target) = &mut node.val {
            *target = new_index[*target];
        }

        new_nodes.push(node);
    }

    Ok(new_nodes)
}

/// Remaps the indices of kernels from node IDs to kernel IDs
///
/// Use this before [`topo_sort`]
pub fn remap_kernels<K>(
    kernels: &mut [Dependencies<Redirect<(K, NodeId, &[bool])>>],
) -> Vec<NodeId> {
    fn node_id<K, F, C>(
        val: &Redirect<(K, NodeId, &[bool])>,
        kernels: &[Dependencies<Redirect<(K, NodeId, &[bool])>>],
        f: F,
    ) -> Result<C, Error>
    where 
        F: FnOnce(&(K, NodeId, &[bool])) -> C
    {
        match val {
            Redirect::Unmasked(val) => Ok(f(val)),
            Redirect::Redirected(idx) => match &kernels[*idx].val {
                Redirect::Unmasked(val) => Ok(f(val)),
                Redirect::Redirected(_) => Err(Error {
                    msg: "double redirection or loop encountered in kernel resolution",
                    kind: ErrorKind::UnresolvedRedirection,
                    ctx: (),
                }),
            },
        }
    }

    let node_to_kernel = kernels
        .iter()
        .enumerate()
        .filter_map(|(idx, dep)| node_id(&dep.val, kernels, |tuple| (tuple.1, idx)).ok())
        .collect::<HashMap<_, _>>();

    let kernel_to_node = kernels
        .iter()
        .filter_map(|dep| node_id(&dep.val, kernels, |(_, node_id, _)| *node_id).ok())
        .collect();

    for dep in kernels {
        for dep in &mut dep.dep {
            *dep = node_to_kernel[&NodeId::from(*dep)];
        }
    }

    kernel_to_node
}

pub fn restore_kernels<K>(
    kernels: &mut [Dependencies<Redirect<(K, NodeId, &[bool])>>],
    kernel_to_node: &[NodeId],
) {
    for dep in kernels {
        for dep in &mut dep.dep {
            *dep = kernel_to_node[*dep];
        }
    }
}

/// Iterates over kernels, calling `f` for each kernel in dep order
///
/// If previously remapped, you must call [`restore_kernels`] first.
pub fn eval_dependency_order<K, F>(
    kernels: &[Dependencies<Redirect<(K, NodeId, &[bool])>>],
    mut f: F,
) -> Result<(), Error>
where
    F: FnMut(&K, NodeId, &[bool]) -> Result<(), Error>,
{
    let mut resolved = Vec::new();
    let mut tmp_res = Vec::new();

    while resolved.len() < kernels.len() {
        for kernel in kernels {
            let dep = &kernel.dep;

            let (kernel, idx, params) = match &kernel.val {
                Redirect::Unmasked(kernel_data) => kernel_data,
                Redirect::Redirected(idx) => match &kernels[*idx].val {
                    Redirect::Unmasked(kernel) => kernel,
                    Redirect::Redirected(_) => {
                        return Err(Error {
                            msg: "double redirection or loop encountered in kernel resolution",
                            kind: ErrorKind::UnresolvedRedirection,
                            ctx: (),
                        });
                    }
                },
            };

            if resolved.contains(idx) {
                continue;
            }

            if dep.iter().all(|x| resolved.contains(x)) {
                tmp_res.push(*idx);

                f(kernel, *idx, params)?;
            }
        }

        resolved.append(&mut tmp_res);
    }

    Ok(())
}
