package iradix

import (
	"bytes"
	"sort"
)

// WalkFn is used when walking the tree.
// Takes a key and value, returning if iteration should be terminated.
type WalkFn func(k []byte, v interface{}) bool

// leafNode is used to represent a value
type leafNode struct {
	mutateCh chan struct{}	// 变更通知管道
	key      []byte			// 键
	val      interface{}	// 值
}

// edge is used to represent an edge node
type edge struct {
	label byte		// 标签
	node  *Node		// 节点
}

// Node is an immutable node in the radix tree
//
// Node 是基数树中的不可变节点
type Node struct {

	// mutateCh is closed if this node is modified
	// 变更通知管道
	mutateCh chan struct{}

	// leaf is used to store possible leaf
	// 叶节点
	leaf *leafNode

	// prefix is the common prefix we ignore
	// 公共前缀
	prefix []byte

	// Edges should be stored in-order for iteration.
	// We avoid a fully materialized slice to save memory,
	// since in most cases we expect to be sparse
	//
	// edges 应按迭代顺序存储。
	// 为了节省内存，我们避免使用 fully materialized 的切片，因为多数情况下切片是稀疏的。
	edges edges
}

// 是否为叶节点
func (n *Node) isLeaf() bool {
	return n.leaf != nil
}

// 添加边，若存在则覆盖
func (n *Node) addEdge(e edge) {
	// 边总数
	num := len(n.edges)
	// 根据 label 二分查找，判断是否已经存在
	idx := sort.Search(num, func(i int) bool {
		return n.edges[i].label >= e.label
	})
	// 添加边
	n.edges = append(n.edges, e)
	// 如果边已经存在，则更新该边
	if idx != num {
		// 移除 n.edges[idx]
		copy(n.edges[idx+1:], n.edges[idx:num])
		// 更新 n.edges[idx]
		n.edges[idx] = e
	}
}

// 替换边
func (n *Node) replaceEdge(e edge) {
	// 查找边
	num := len(n.edges)
	idx := sort.Search(num, func(i int) bool {
		return n.edges[i].label >= e.label
	})
	// 如果边已经存在，则更新
	if idx < num && n.edges[idx].label == e.label {
		n.edges[idx].node = e.node
		return
	}
	// 如果边不存在，panic
	panic("replacing missing edge")
}

// 查找边，返回下标和子节点
func (n *Node) getEdge(label byte) (int, *Node) {
	num := len(n.edges)
	idx := sort.Search(num, func(i int) bool {
		return n.edges[i].label >= label
	})
	if idx < num && n.edges[idx].label == label {
		return idx, n.edges[idx].node
	}
	return -1, nil
}

func (n *Node) getLowerBoundEdge(label byte) (int, *Node) {
	num := len(n.edges)
	idx := sort.Search(num, func(i int) bool {
		return n.edges[i].label >= label
	})
	// we want lower bound behavior so return even if it's not an exact match
	if idx < num {
		return idx, n.edges[idx].node
	}
	return -1, nil
}


// 删除边
func (n *Node) delEdge(label byte) {
	num := len(n.edges)
	idx := sort.Search(num, func(i int) bool {
		return n.edges[i].label >= label
	})
	if idx < num && n.edges[idx].label == label {
		copy(n.edges[idx:], n.edges[idx+1:])
		n.edges[len(n.edges)-1] = edge{}
		n.edges = n.edges[:len(n.edges)-1]
	}
}

// GetWatch xxx
func (n *Node) GetWatch(k []byte) (<-chan struct{}, interface{}, bool) {
	search := k
	watch := n.mutateCh

	for {
		// Check for key exhaustion
		if len(search) == 0 {
			// 如果当前节点为叶节点，就直接返回 `通知管道、值、true`
			if n.isLeaf() {
				return n.leaf.mutateCh, n.leaf.val, true
			}
			// 未找到
			break
		}

		// Look for an edge
		// 在当前节点中查找边，返回下标(_)和边节点
		_, n = n.getEdge(search[0])
		if n == nil {
			// 未找到
			break
		}

		// Update to the finest granularity as the search makes progress
		// 随着搜索的进行，更新到最精细的粒度
		watch = n.mutateCh

		// Consume the search prefix
		// 移除节点前缀
		if bytes.HasPrefix(search, n.prefix) {
			search = search[len(n.prefix):]
		} else {
			// 未找到
			break
		}
	}

	return watch, nil, false
}

func (n *Node) Get(k []byte) (interface{}, bool) {
	_, val, ok := n.GetWatch(k)
	return val, ok
}

// LongestPrefix is like Get, but instead of an
// exact match, it will return the longest prefix match.
func (n *Node) LongestPrefix(k []byte) ([]byte, interface{}, bool) {
	var last *leafNode
	search := k
	for {

		// Look for a leaf node
		if n.isLeaf() {
			last = n.leaf
		}

		// Check for key exhaution
		if len(search) == 0 {
			break
		}

		// Look for an edge
		_, n = n.getEdge(search[0])
		if n == nil {
			break
		}

		// Consume the search prefix
		if bytes.HasPrefix(search, n.prefix) {
			search = search[len(n.prefix):]
		} else {
			break
		}
	}
	if last != nil {
		return last.key, last.val, true
	}
	return nil, nil, false
}

// Minimum is used to return the minimum value in the tree
func (n *Node) Minimum() ([]byte, interface{}, bool) {
	for {
		if n.isLeaf() {
			return n.leaf.key, n.leaf.val, true
		}
		if len(n.edges) > 0 {
			n = n.edges[0].node
		} else {
			break
		}
	}
	return nil, nil, false
}

// Maximum is used to return the maximum value in the tree
func (n *Node) Maximum() ([]byte, interface{}, bool) {
	for {
		if num := len(n.edges); num > 0 {
			n = n.edges[num-1].node
			continue
		}
		if n.isLeaf() {
			return n.leaf.key, n.leaf.val, true
		} else {
			break
		}
	}
	return nil, nil, false
}

// Iterator is used to return an iterator at
// the given node to walk the tree
func (n *Node) Iterator() *Iterator {
	return &Iterator{node: n}
}

// ReverseIterator is used to return an iterator at
// the given node to walk the tree backwards
func (n *Node) ReverseIterator() *ReverseIterator {
	return NewReverseIterator(n)
}

// rawIterator is used to return a raw iterator at the given node to walk the
// tree.
func (n *Node) rawIterator() *rawIterator {
	iter := &rawIterator{node: n}
	iter.Next()
	return iter
}

// Walk is used to walk the tree
func (n *Node) Walk(fn WalkFn) {
	recursiveWalk(n, fn)
}

// WalkBackwards is used to walk the tree in reverse order
func (n *Node) WalkBackwards(fn WalkFn) {
	reverseRecursiveWalk(n, fn)
}

// WalkPrefix is used to walk the tree under a prefix
func (n *Node) WalkPrefix(prefix []byte, fn WalkFn) {
	search := prefix
	for {
		// Check for key exhaution
		if len(search) == 0 {
			recursiveWalk(n, fn)
			return
		}

		// Look for an edge
		_, n = n.getEdge(search[0])
		if n == nil {
			break
		}

		// Consume the search prefix
		if bytes.HasPrefix(search, n.prefix) {
			search = search[len(n.prefix):]

		} else if bytes.HasPrefix(n.prefix, search) {
			// Child may be under our search prefix
			recursiveWalk(n, fn)
			return
		} else {
			break
		}
	}
}

// WalkPath is used to walk the tree, but only visiting nodes
// from the root down to a given leaf. Where WalkPrefix walks
// all the entries *under* the given prefix, this walks the
// entries *above* the given prefix.
func (n *Node) WalkPath(path []byte, fn WalkFn) {
	search := path
	for {
		// Visit the leaf values if any
		if n.leaf != nil && fn(n.leaf.key, n.leaf.val) {
			return
		}

		// Check for key exhaution
		if len(search) == 0 {
			return
		}

		// Look for an edge
		_, n = n.getEdge(search[0])
		if n == nil {
			return
		}

		// Consume the search prefix
		if bytes.HasPrefix(search, n.prefix) {
			search = search[len(n.prefix):]
		} else {
			break
		}
	}
}

// recursiveWalk is used to do a pre-order walk of a node
// recursively. Returns true if the walk should be aborted
func recursiveWalk(n *Node, fn WalkFn) bool {
	// Visit the leaf values if any
	if n.leaf != nil && fn(n.leaf.key, n.leaf.val) {
		return true
	}

	// Recurse on the children
	for _, e := range n.edges {
		if recursiveWalk(e.node, fn) {
			return true
		}
	}
	return false
}

// reverseRecursiveWalk is used to do a reverse pre-order
// walk of a node recursively. Returns true if the walk
// should be aborted
func reverseRecursiveWalk(n *Node, fn WalkFn) bool {
	// Visit the leaf values if any
	if n.leaf != nil && fn(n.leaf.key, n.leaf.val) {
		return true
	}

	// Recurse on the children in reverse order
	for i := len(n.edges) - 1; i >= 0; i-- {
		e := n.edges[i]
		if reverseRecursiveWalk(e.node, fn) {
			return true
		}
	}
	return false
}
