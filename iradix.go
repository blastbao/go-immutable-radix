package iradix

import (
	"bytes"
	"strings"

	"github.com/hashicorp/golang-lru/simplelru"
)

const (
	// defaultModifiedCache is the default size of the modified node
	// cache used per transaction. This is used to cache the updates
	// to the nodes near the root, while the leaves do not need to be
	// cached. This is important for very large transactions to prevent
	// the modified cache from growing to be enormous. This is also used
	// to set the max size of the mutation notify maps since those should
	// also be bounded in a similar way.
	//
	// defaultModifiedCache 是每个事务使用的修改节点缓存的默认大小。
	// 这用于缓存对根节点附近节点的更新，而叶节点不需要缓存。
	// 这对于大事务非常重要，可以防止修改后的缓存变得巨大。
	// 这也用于设置 mutation 通知 maps 的最大大小，因为这些 maps 也应该以类似的方式进行绑定。
	//
	defaultModifiedCache = 8192
)

// Tree implements an immutable radix tree. This can be treated as a
// Dictionary abstract data type. The main advantage over a standard
// hash map is prefix-based lookups and ordered iteration. The immutability
// means that it is safe to concurrently read from a Tree without any
// coordination.
//
// Tree 实现了一个不可变的基数树。 可以将 Tree 视为字典，与标准 HashMap 相比，
// 它的主要优点是基于前缀的查找和有序迭代。不变性意味着无需任何协调（锁）就可以安全地并发读取。
//
type Tree struct {
	root *Node	// 根节点
	size int	// 节点数
}

// New returns an empty Tree
func New() *Tree {
	t := &Tree{
		root: &Node{
			mutateCh: make(chan struct{}),
		},
	}
	return t
}

// Len is used to return the number of elements in the tree
func (t *Tree) Len() int {
	return t.size
}

// Txn is a transaction on the tree.
//
// This transaction is applied atomically and returns a new tree when committed.
// A transaction is not thread safe, and should only be used by a single goroutine.
//
// 事务以原子方式执行，并在提交时返回一个新树。
// 事务不是线程安全的，只能由单个 goroutine 使用。
type Txn struct {

	// root is the modified root for the transaction.
	//
	// root 是事务的根节点。
	root *Node

	// snap is a snapshot of the root node for use if we have to run the
	// slow notify algorithm.
	//
	// snap 是根节点的快照，如果必须运行 slow notify 算法，可以使用它。
	snap *Node

	// size tracks the size of the tree as it is modified during the
	// transaction.
	//
	// size 跟踪在事务期间修改的树的大小。
	size int

	// writable is a cache of writable nodes that have been created during
	// the course of the transaction. This allows us to re-use the same
	// nodes for further writes and avoid unnecessary copies of nodes that
	// have never been exposed outside the transaction. This will only hold
	// up to defaultModifiedCache number of entries.
	//
	// writable 是在事务过程中创建的可写节点的缓存，以允许我们重新使用相同的节点进行进
	// 一步的写入，并避免不必要的拷贝那些未在事务外部公开的节点。
	//
	// 最多只能容纳 defaultModifiedCache 条目数。
	writable *simplelru.LRU

	// trackChannels is used to hold channels that need to be notified to
	// signal mutation of the tree. This will only hold up to
	// defaultModifiedCache number of entries, after which we will set the
	// trackOverflow flag, which will cause us to use a more expensive
	// algorithm to perform the notifications. Mutation tracking is only
	// performed if trackMutate is true.
	//
	// trackChannels 用于保存需要通知的管道，以通知树的变更。
	//
	// trackChannels 最多只能容纳 defaultModifiedCache 数量的条目，
	// 之后我们将设置 trackOverflow 标志，这将导致我们使用更昂贵的算法来执行通知。
	// 只有 trackMutate 为 true 时才执行变更跟踪。
	trackChannels map[chan struct{}]struct{}
	trackOverflow bool
	trackMutate   bool
}

// Txn starts a new transaction that can be used to mutate the tree
//
// Txn 启动一个新事务，该事务可用于修改树。
func (t *Tree) Txn() *Txn {
	txn := &Txn{
		root: t.root,
		snap: t.root,
		size: t.size,
	}
	return txn
}

// Clone makes an independent copy of the transaction.
// The new transaction does not track any nodes and has TrackMutate turned off.
//
// The cloned transaction will contain any uncommitted writes in the original
// transaction but further mutations to either will be independent and result
// in different radix trees on Commit. A cloned transaction may be passed to
// another goroutine and mutated there independently however each transaction
// may only be mutated in a single thread.
//
//
// Clone 生成事务的独立拷贝。
// 新事务不跟踪任何节点，并且将关闭 TrackMutate 。
//
// 被克隆的事务会包含原始事务中未提交的写操作，但对这两个事务的进一步变更将是独立的，
// 并在提交时生成不同的基数树。
//
// 被克隆克隆的事务可以传递给另一个 goroutine 并在那里独立地进行变更，但是每个事务只能在单个线程中进行变更。
//
func (t *Txn) Clone() *Txn {
	// reset the writable node cache to avoid leaking future writes into the clone
	// 重置可写节点的缓存，以避免将来的写入泄漏到副本中。
	t.writable = nil

	txn := &Txn{
		root: t.root,
		snap: t.snap,
		size: t.size,
	}

	return txn
}

// TrackMutate can be used to toggle if mutations are tracked. If this is enabled
// then notifications will be issued for affected internal nodes and leaves when
// the transaction is committed.
//
// TrackMutate 设置是否在跟踪变更时。
func (t *Txn) TrackMutate(track bool) {
	t.trackMutate = track
}

// trackChannel safely attempts to track the given mutation channel, setting the
// overflow flag if we can no longer track any more. This limits the amount of
// state that will accumulate during a transaction and we have a slower algorithm
// to switch to if we overflow.
func (t *Txn) trackChannel(ch chan struct{}) {
	// In overflow, make sure we don't store any more objects.
	if t.trackOverflow {
		return
	}

	// If this would overflow the state we reject it and set the flag (since
	// we aren't tracking everything that's required any longer).
	if len(t.trackChannels) >= defaultModifiedCache {
		// Mark that we are in the overflow state
		t.trackOverflow = true

		// Clear the map so that the channels can be garbage collected. It is
		// safe to do this since we have already overflowed and will be using
		// the slow notify algorithm.
		t.trackChannels = nil
		return
	}

	// Create the map on the fly when we need it.
	if t.trackChannels == nil {
		t.trackChannels = make(map[chan struct{}]struct{})
	}

	// Otherwise we are good to track it.
	t.trackChannels[ch] = struct{}{}
}

// writeNode returns a node to be modified, if the current node has already been
// modified during the course of the transaction, it is used in-place.
//
// Set forLeafUpdate to true if you are getting a write node to update the leaf,
// which will set leaf mutation tracking appropriately as well.
//
// writeNode 返回要修改的节点，如果当前节点在事务处理过程中已被修改，则它将被就地使用。
//
// 如果要获得一个写入节点来更新 leaf ，请将 forLeafUpdate 设置为 true ，这也将适当地设置 leaf mutation tracking 。
func (t *Txn) writeNode(n *Node, forLeafUpdate bool) *Node {

	// Ensure the writable set exists.
	//
	// 初始化节点缓存(LRU)
	if t.writable == nil {
		lru, err := simplelru.NewLRU(defaultModifiedCache, nil)
		if err != nil {
			panic(err)
		}
		t.writable = lru
	}

	// If this node has already been modified, we can continue to use it
	// during this transaction. We know that we don't need to track it for
	// a node update since the node is writable, but if this is for a leaf
	// update we track it, in case the initial write to this node didn't
	// update the leaf.
	//
	// 如果节点 n 已被修改，我们可以在此事务期间继续使用它。
	if _, ok := t.writable.Get(n); ok {
		// 如果：
		// 	1. 开启了变更追踪
		// 	2. 更新 leaf
		// 	3. 节点 leaf 非空
		// 则：
		//
		if t.trackMutate && forLeafUpdate && n.leaf != nil {
			t.trackChannel(n.leaf.mutateCh)
		}

		return n
	}

	// Mark this node as being mutated.
	if t.trackMutate {
		t.trackChannel(n.mutateCh)
	}

	// Mark its leaf as being mutated, if appropriate.
	if t.trackMutate && forLeafUpdate && n.leaf != nil {
		t.trackChannel(n.leaf.mutateCh)
	}

	// Copy the existing node. If you have set forLeafUpdate it will be
	// safe to replace this leaf with another after you get your node for
	// writing. You MUST replace it, because the channel associated with
	// this leaf will be closed when this transaction is committed.
	//
	// 复制现有节点。
	//
	// 如果您已设置为 LeafUpdate ，则在获取节点进行写入后，可以安全地将此叶替换为另一个叶。
	// 您必须替换它，因为提交此事务时，与此叶关联的通道将关闭。
	//
	nc := &Node{
		mutateCh: make(chan struct{}),
		leaf:     n.leaf,
	}

	// 把 n.prefix 拷贝到 nc.prefix
	if n.prefix != nil {
		nc.prefix = make([]byte, len(n.prefix))
		copy(nc.prefix, n.prefix)
	}

	// 把 n.edges 拷贝到 nc.edges
	if len(n.edges) != 0 {
		nc.edges = make([]edge, len(n.edges))
		copy(nc.edges, n.edges)
	}

	// Mark this node as writable.
	// 将节点 nc 保存到缓存中
	t.writable.Add(nc, nil)
	return nc
}

// Visit all the nodes in the tree under n, and add their mutateChannels to the transaction
// Returns the size of the subtree visited
func (t *Txn) trackChannelsAndCount(n *Node) int {
	// Count only leaf nodes
	leaves := 0
	if n.leaf != nil {
		leaves = 1
	}
	// Mark this node as being mutated.
	if t.trackMutate {
		t.trackChannel(n.mutateCh)
	}

	// Mark its leaf as being mutated, if appropriate.
	if t.trackMutate && n.leaf != nil {
		t.trackChannel(n.leaf.mutateCh)
	}

	// Recurse on the children
	for _, e := range n.edges {
		leaves += t.trackChannelsAndCount(e.node)
	}
	return leaves
}

// mergeChild is called to collapse the given node with its child.
// This is only called when the given node is not a leaf and has a single edge.
func (t *Txn) mergeChild(n *Node) {
	// Mark the child node as being mutated since we are about to abandon it.
	// We don't need to mark the leaf since we are retaining it if it is there.

	// 取出唯一子节点
	e := n.edges[0]
	child := e.node

	if t.trackMutate {
		t.trackChannel(child.mutateCh)
	}

	// Merge the nodes.

	// 合并子节点的 prefix
	n.prefix = concat(n.prefix, child.prefix)
	// 合并子节点的 leaf
	n.leaf = child.leaf
	// 合并子节点的 edges
	if len(child.edges) != 0 {
		n.edges = make([]edge, len(child.edges))
		copy(n.edges, child.edges)
	} else {
		n.edges = nil
	}
}

// insert does a recursive insertion
//
//
// 假设 search = "abcdef"
//
func (t *Txn) insert(n *Node, k, search []byte, v interface{}) (*Node, interface{}, bool) {

	// Handle key exhaustion
	if len(search) == 0 {

		// 默认旧值为空，update 为 false
		var oldVal interface{}
		didUpdate := false

		// 如果 n 是叶节点，意味着其上已经有值，就取出其旧值，并设置 update 为 true
		if n.isLeaf() {
			oldVal = n.leaf.val
			didUpdate = true
		}

		// 构造 n 节点的拷贝 nc
		nc := t.writeNode(n, true)
		// 更新 nc 的叶节点
		nc.leaf = &leafNode{
			mutateCh: make(chan struct{}),
			key:      k,
			val:      v,
		}
		return nc, oldVal, didUpdate
	}

	// Look for the edge
	// 按字符查询 node 下的边，search[0] == "a"
	idx, child := n.getEdge(search[0])

	// No edge, create one
	// 不存在，则新建一条边，其关联了一个叶节点
	if child == nil {
		// 构造边
		e := edge{
			// 标签
			label: search[0],	// "a"
			// 边节点
			node: &Node{
				mutateCh: make(chan struct{}),		// 通知管道
				leaf: &leafNode{					// 叶节点
					mutateCh: make(chan struct{}),	// 通知管道
					key:      k,					// key
					val:      v,					// value
				},
				prefix: search,						// "abcdef"
			},
		}
		// 构造 n 节点的拷贝 nc
		nc := t.writeNode(n, false)
		// 为 nc 添加边 e
		nc.addEdge(e)
		return nc, nil, false
	}

	// Determine longest prefix of the search key on match

	// 获取两个 []byte 的最长公共前缀长度，
	// eg.
	// 	Case 1: abcdef 和 abcd 的公共前缀长度为 4 。
	// 	Case 2: abcdef 和 abchmx 的公共前缀长度为 3 。
	commonPrefix := longestPrefix(search, child.prefix)

	// Case 1:
	// 	如果 search 串包含 child.prefix 子串，就可以在 child 下创建一个子节点，其 search = abcdefg - abcd = efg 。
	if commonPrefix == len(child.prefix) {

		// search = abcdefg - abcd = efg
		search = search[commonPrefix:]

		// [递归] 把 "efg" 插入到子节点 child 中
		newChild, oldVal, didUpdate := t.insert(child, k, search, v)

		//
		if newChild != nil {
			// 创建 n 节点的拷贝 nc
			nc := t.writeNode(n, false)
			// 覆盖更新节点 nc 的第 idx 个子节点
			nc.edges[idx].node = newChild
			// 返回新节点、原始值、是否更新
			return nc, oldVal, didUpdate
		}

		//
		return nil, oldVal, didUpdate
	}

	// Case 2: 需要分裂节点

	// Split the node

	// 创建节点 n 的拷贝 nc
	nc := t.writeNode(n, false)

	// 创建分裂节点
	splitNode := &Node{
		mutateCh: make(chan struct{}),
		prefix:   search[:commonPrefix], // prefix = xxx(abcdef,abchmx) = abc
	}

	// 分裂节点作为 nc 的子节点
	nc.replaceEdge(edge{
		label: search[0], 	// label = "a"
		node:  splitNode,	//
	})

	// Restore the existing child node
	//
	// 创建 child 节点的拷贝 modChild
	modChild := t.writeNode(child, false)

	// 为分裂节点创建子节点
	splitNode.addEdge(edge{
		label: modChild.prefix[commonPrefix],	// abcd[3] = "h"
		node:  modChild,						//
	})

	modChild.prefix = modChild.prefix[commonPrefix:] // abcd[3:] = "hmx"

	// Create a new leaf node
	leaf := &leafNode{
		mutateCh: make(chan struct{}),
		key:      k,
		val:      v,
	}

	// If the new key is a subset, add to to this node
	search = search[commonPrefix:]	// search[commonPrefix:] = abcdef[3:] = "def"

	// 如果 len(search) == 0 ，则分裂节点本身是叶子节点，就把 leaf 保存到其上
	if len(search) == 0 {
		splitNode.leaf = leaf
		return nc, nil, false
	}

	// 否则，分裂节点非子节点，为其添加一条边

	// Create a new edge for the node
	splitNode.addEdge(edge{
		label: search[0],		// "d"
		node: &Node{
			mutateCh: make(chan struct{}),	// 通知管道
			leaf:     leaf,					// 叶节点
			prefix:   search,				// 前缀: "def"
		},
	})

	return nc, nil, false
}

// delete does a recursive deletion
func (t *Txn) delete(parent, n *Node, search []byte) (*Node, *leafNode) {

	// Check for key exhaustion
	if len(search) == 0 {

		// 如果 n 非叶节点，什么都不用做
		if !n.isLeaf() {
			return nil, nil
		}

		// 如果 n 为叶节点，需要将其 leaf 字段重置，如果此时 n 有且仅有一个子节点，则可以合并二者。

		// Copy the pointer in case we are in a transaction that already
		// modified this node since the node will be reused. Any changes
		// made to the node will not affect returning the original leaf
		// value.
		//
		oldLeaf := n.leaf

		// Remove the leaf node
		//
		// 创建节点 n 的拷贝，并重置其 leaf ，相当于执行删除操作
		nc := t.writeNode(n, true)
		nc.leaf = nil

		// Check if this node should be merged
		//
		// 如果 n 非根节点，且有且仅有一个子节点，就合并这个子节点。
		if n != t.root && len(nc.edges) == 1 {
			t.mergeChild(nc)
		}

		//
		return nc, oldLeaf
	}

	// Look for an edge
	label := search[0]
	idx, child := n.getEdge(label)

	// 不存在，返回 nil,nil
	if child == nil || !bytes.HasPrefix(search, child.prefix) {
		return nil, nil
	}

	// Consume the search prefix
	search = search[len(child.prefix):]
	newChild, leaf := t.delete(n, child, search)

	// 不存在，返回 nil,nil
	if newChild == nil {
		return nil, nil
	}

	// Copy this node. WATCH OUT - it's safe to pass "false" here because we
	// will only ADD a leaf via nc.mergeChild() if there isn't one due to
	// the !nc.isLeaf() check in the logic just below. This is pretty subtle,
	// so be careful if you change any of the logic here.
	//
	// 创建 n 的拷贝
	nc := t.writeNode(n, false)

	// Delete the edge if the node has no edges
	//
	// 如果 newChild 非叶节点，且其子节点为空，则可以从 nc 中删除这个子节点。
	if newChild.leaf == nil && len(newChild.edges) == 0 {
		nc.delEdge(label)
		if n != t.root && len(nc.edges) == 1 && !nc.isLeaf() {
			t.mergeChild(nc)
		}
	// 否则，就用 newChild 覆盖旧的边
	} else {
		nc.edges[idx].node = newChild
	}

	return nc, leaf
}

// delete does a recursive deletion
func (t *Txn) deletePrefix(parent, n *Node, search []byte) (*Node, int) {
	// Check for key exhaustion
	if len(search) == 0 {
		nc := t.writeNode(n, true)
		if n.isLeaf() {
			nc.leaf = nil
		}
		nc.edges = nil
		return nc, t.trackChannelsAndCount(n)
	}

	// Look for an edge
	label := search[0]
	idx, child := n.getEdge(label)
	// We make sure that either the child node's prefix starts with the search term, or the search term starts with the child node's prefix
	// Need to do both so that we can delete prefixes that don't correspond to any node in the tree
	if child == nil || (!bytes.HasPrefix(child.prefix, search) && !bytes.HasPrefix(search, child.prefix)) {
		return nil, 0
	}

	// Consume the search prefix
	if len(child.prefix) > len(search) {
		search = []byte("")
	} else {
		search = search[len(child.prefix):]
	}
	newChild, numDeletions := t.deletePrefix(n, child, search)
	if newChild == nil {
		return nil, 0
	}
	// Copy this node. WATCH OUT - it's safe to pass "false" here because we
	// will only ADD a leaf via nc.mergeChild() if there isn't one due to
	// the !nc.isLeaf() check in the logic just below. This is pretty subtle,
	// so be careful if you change any of the logic here.

	nc := t.writeNode(n, false)

	// Delete the edge if the node has no edges
	if newChild.leaf == nil && len(newChild.edges) == 0 {
		nc.delEdge(label)
		if n != t.root && len(nc.edges) == 1 && !nc.isLeaf() {
			t.mergeChild(nc)
		}
	} else {
		nc.edges[idx].node = newChild
	}
	return nc, numDeletions
}

// Insert is used to add or update a given key.
// The return provides the previous value and a bool indicating if any was set.
func (t *Txn) Insert(k []byte, v interface{}) (interface{}, bool) {
	// 执行插入
	newRoot, oldVal, didUpdate := t.insert(t.root, k, k, v)

	// 替换根节点
	if newRoot != nil {
		t.root = newRoot
	}

	// 如果为新增，则增加数据条数
	if !didUpdate {
		t.size++
	}

	// 返回旧值、是否更新
	return oldVal, didUpdate
}

// Delete is used to delete a given key.
// Returns the old value if any, and a bool indicating if the key was set.
func (t *Txn) Delete(k []byte) (interface{}, bool) {

	newRoot, leaf := t.delete(nil, t.root, k)

	// 替换根节点
	if newRoot != nil {
		t.root = newRoot
	}

	// 删除了叶结点，减少数据条数
	if leaf != nil {
		t.size--
		return leaf.val, true
	}

	return nil, false
}

// DeletePrefix is used to delete an entire subtree that matches the prefix
// This will delete all nodes under that prefix
func (t *Txn) DeletePrefix(prefix []byte) bool {
	newRoot, numDeletions := t.deletePrefix(nil, t.root, prefix)
	if newRoot != nil {
		t.root = newRoot
		t.size = t.size - numDeletions
		return true
	}
	return false

}

// Root returns the current root of the radix tree within this
// transaction. The root is not safe across insert and delete operations,
// but can be used to read the current state during a transaction.
func (t *Txn) Root() *Node {
	return t.root
}

// Get is used to lookup a specific key, returning
// the value and if it was found
func (t *Txn) Get(k []byte) (interface{}, bool) {
	return t.root.Get(k)
}

// GetWatch is used to lookup a specific key, returning
// the watch channel, value and if it was found
func (t *Txn) GetWatch(k []byte) (<-chan struct{}, interface{}, bool) {
	return t.root.GetWatch(k)
}

// Commit is used to finalize the transaction and return a new tree. If mutation
// tracking is turned on then notifications will also be issued.
func (t *Txn) Commit() *Tree {
	nt := t.CommitOnly()
	if t.trackMutate {
		t.Notify()
	}
	return nt
}

// CommitOnly is used to finalize the transaction and return a new tree, but
// does not issue any notifications until Notify is called.
func (t *Txn) CommitOnly() *Tree {
	nt := &Tree{t.root, t.size}
	t.writable = nil
	return nt
}

// slowNotify does a complete comparison of the before and after trees in order
// to trigger notifications. This doesn't require any additional state but it
// is very expensive to compute.
func (t *Txn) slowNotify() {
	snapIter := t.snap.rawIterator()
	rootIter := t.root.rawIterator()
	for snapIter.Front() != nil || rootIter.Front() != nil {
		// If we've exhausted the nodes in the old snapshot, we know
		// there's nothing remaining to notify.
		if snapIter.Front() == nil {
			return
		}
		snapElem := snapIter.Front()

		// If we've exhausted the nodes in the new root, we know we need
		// to invalidate everything that remains in the old snapshot. We
		// know from the loop condition there's something in the old
		// snapshot.
		if rootIter.Front() == nil {
			close(snapElem.mutateCh)
			if snapElem.isLeaf() {
				close(snapElem.leaf.mutateCh)
			}
			snapIter.Next()
			continue
		}

		// Do one string compare so we can check the various conditions
		// below without repeating the compare.
		cmp := strings.Compare(snapIter.Path(), rootIter.Path())

		// If the snapshot is behind the root, then we must have deleted
		// this node during the transaction.
		if cmp < 0 {
			close(snapElem.mutateCh)
			if snapElem.isLeaf() {
				close(snapElem.leaf.mutateCh)
			}
			snapIter.Next()
			continue
		}

		// If the snapshot is ahead of the root, then we must have added
		// this node during the transaction.
		if cmp > 0 {
			rootIter.Next()
			continue
		}

		// If we have the same path, then we need to see if we mutated a
		// node and possibly the leaf.
		rootElem := rootIter.Front()
		if snapElem != rootElem {
			close(snapElem.mutateCh)
			if snapElem.leaf != nil && (snapElem.leaf != rootElem.leaf) {
				close(snapElem.leaf.mutateCh)
			}
		}
		snapIter.Next()
		rootIter.Next()
	}
}

// Notify is used along with TrackMutate to trigger notifications. This must
// only be done once a transaction is committed via CommitOnly, and it is called
// automatically by Commit.
func (t *Txn) Notify() {
	if !t.trackMutate {
		return
	}

	// If we've overflowed the tracking state we can't use it in any way and
	// need to do a full tree compare.
	if t.trackOverflow {
		t.slowNotify()
	} else {
		for ch := range t.trackChannels {
			close(ch)
		}
	}

	// Clean up the tracking state so that a re-notify is safe (will trigger
	// the else clause above which will be a no-op).
	t.trackChannels = nil
	t.trackOverflow = false
}

// Insert is used to add or update a given key. The return provides
// the new tree, previous value and a bool indicating if any was set.
func (t *Tree) Insert(k []byte, v interface{}) (*Tree, interface{}, bool) {
	txn := t.Txn()
	old, ok := txn.Insert(k, v)
	return txn.Commit(), old, ok
}

// Delete is used to delete a given key. Returns the new tree,
// old value if any, and a bool indicating if the key was set.
func (t *Tree) Delete(k []byte) (*Tree, interface{}, bool) {
	txn := t.Txn()
	old, ok := txn.Delete(k)
	return txn.Commit(), old, ok
}

// DeletePrefix is used to delete all nodes starting with a given prefix. Returns the new tree,
// and a bool indicating if the prefix matched any nodes
func (t *Tree) DeletePrefix(k []byte) (*Tree, bool) {
	txn := t.Txn()
	ok := txn.DeletePrefix(k)
	return txn.Commit(), ok
}

// Root returns the root node of the tree which can be used for richer
// query operations.
func (t *Tree) Root() *Node {
	return t.root
}

// Get is used to lookup a specific key, returning
// the value and if it was found
func (t *Tree) Get(k []byte) (interface{}, bool) {
	return t.root.Get(k)
}

// longestPrefix finds the length of the shared prefix of two strings
//
// longestPrefix 返回两个 []byte 的最长公共前缀长度。
func longestPrefix(k1, k2 []byte) int {
	// max(len(k1), len(k2)
	max := len(k1)
	if l := len(k2); l < max {
		max = l
	}

	// 找到首个不同的字符的下标
	var i int
	for i = 0; i < max; i++ {
		if k1[i] != k2[i] {
			break
		}
	}
	return i
}

// concat two byte slices, returning a third new copy
func concat(a, b []byte) []byte {
	c := make([]byte, len(a)+len(b))
	copy(c, a)
	copy(c[len(a):], b)
	return c
}
