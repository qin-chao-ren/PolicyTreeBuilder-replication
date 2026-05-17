from typing import Dict, List, Optional, Any, Set


class TreeManager:
    """
    v4 动态树拓扑管理器
    功能：
    1. 统一管理节点索引，防止 Stale Index（过期索引）。
    2. 强制清洗 tree_id，只保留 node_id。
    3. 提供原子操作（Move, Remove, Promote），并维护父子一致性。
    """
    def __init__(self, root: Dict[str, Any]):
        self.root = root
        self.index: Dict[str, Dict] = {}        # node_id -> node_obj
        self.parent_map: Dict[str, str] = {}    # node_id -> parent_id
        self._build_index(self.root, None)

    def _build_index(self, node: Dict, parent_id: Optional[str]):
        """递归构建索引，并清洗脏数据"""
        # 1. 统一 ID：优先取 node_id，清洗 tree_id
        nid = str(node.get("node_id") or node.get("tree_id"))
        if not nid:
            raise ValueError(f"节点缺少 ID: {node}")

        node["node_id"] = nid
        # 【关键】彻底移除 tree_id，防止分裂
        if "tree_id" in node:
            del node["tree_id"]

        # 2. 注册
        self.index[nid] = node
        if parent_id:
            self.parent_map[nid] = parent_id

        # 3. 递归处理子节点
        children = node.get("children", [])
        if not children:
            node["children"] = []  # 确保列表存在

        for ch in list(children):  # 使用 list 副本防止遍历时修改
            self._build_index(ch, nid)

    # --- 查询 API ---

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self.index.get(node_id)

    def exists(self, node_id: str) -> bool:
        return node_id in self.index

    def get_parent_id(self, node_id: str) -> Optional[str]:
        return self.parent_map.get(node_id)

    def get_children(self, node_id: str) -> List[Dict]:
        node = self.index.get(node_id)
        return node.get("children", []) if node else []

    def get_all_node_ids(self) -> List[str]:
        return list(self.index.keys())

    # --- 核心原子操作 (Write Ops) ---

    def move_node(self, node_id: str, new_parent_id: str) -> bool:
        """
        将节点移动到新父节点下。
        返回: Success (bool)
        """
        if node_id not in self.index or new_parent_id not in self.index:
            return False

        # 1. 防止自环或移动到子孙节点
        if node_id == new_parent_id:
            return False

        # 2. 从旧父节点移除
        old_parent_id = self.parent_map.get(node_id)
        if old_parent_id:
            old_parent = self.index[old_parent_id]
            old_parent["children"] = [c for c in old_parent["children"] if c["node_id"] != node_id]

        # 3. 挂载到新父节点
        new_parent = self.index[new_parent_id]
        node = self.index[node_id]
        new_parent.setdefault("children", []).append(node)

        # 4. 更新索引
        self.parent_map[node_id] = new_parent_id
        return True

    def remove_node(self, node_id: str, keep_children_orphaned: bool = False) -> bool:
        """
        删除节点。
        keep_children_orphaned:
           False (默认) -> 级联删除（子树全删）
           True -> 子节点保留在 Index 中但无父节点（变为孤儿，慎用）
        """
        if node_id not in self.index:
            return False

        node = self.index[node_id]

        # 1. 如果不保留子节点，递归清理索引
        if not keep_children_orphaned:
            stack = [node]
            while stack:
                curr = stack.pop()
                cid = curr["node_id"]
                if cid in self.index:
                    del self.index[cid]
                if cid in self.parent_map:
                    del self.parent_map[cid]
                stack.extend(curr.get("children", []))

        # 2. 从父节点引用中移除
        parent_id = self.parent_map.get(node_id)
        if parent_id and parent_id in self.index:
            parent = self.index[parent_id]
            parent["children"] = [c for c in parent["children"] if c["node_id"] != node_id]

        return True

    def promote_child_safe(self, child_id: str) -> bool:
        """
        【Fail-Safe Promote】
        尝试将 child 提升到 爷爷(Grandparent) 下。
        如果爷爷不存在（即父节点是 Root/L1 或者是孤儿），则拒绝提升，返回 False。
        """
        parent_id = self.get_parent_id(child_id)
        if not parent_id:
            return False

        grand_id = self.get_parent_id(parent_id)
        if not grand_id:
            # 兜底逻辑：没有爷爷，无法提升 (不能挂到虚空，也不能随便挂到 Root)
            return False

        # 执行移动
        return self.move_node(child_id, grand_id)

    def absorb_node(self, winner_id: str, loser_id: str) -> bool:
        """
        Winner 吸收 Loser：
        1. Loser 的所有孩子 -> 移给 Winner
        2. 删除 Loser
        """
        if winner_id not in self.index or loser_id not in self.index:
            return False

        loser_children = list(self.get_children(loser_id))  # 快照
        for child in loser_children:
            self.move_node(child["node_id"], winner_id)

        self.remove_node(loser_id)
        return True
    
    def add_child_node(self, parent_id: str, new_node_data: Dict) -> bool:
        """
        在指定父节点下注册一个全新的节点。
        自动处理索引注册、ID清洗和父子关系挂载。
        """
        if parent_id not in self.index:
            return False
            
        # 1. 清洗与检查 ID
        nid = str(new_node_data.get("node_id") or new_node_data.get("tree_id"))
        if not nid:
            raise ValueError("New node must have an ID")
        if nid in self.index:
            # 防止 ID 冲突，如果有冲突，自动添加随机后缀 (Fail-Safe)
            import uuid
            nid = f"{nid}_{str(uuid.uuid4())[:4]}"
        
        new_node_data["node_id"] = nid
        if "tree_id" in new_node_data:
            del new_node_data["tree_id"]
        
        # 确保 children 列表存在
        new_node_data.setdefault("children", [])
        
        # 2. 注册索引
        self.index[nid] = new_node_data
        self.parent_map[nid] = parent_id
        
        # 3. 挂载到父节点
        parent = self.index[parent_id]
        parent.setdefault("children", []).append(new_node_data)
        
        return True