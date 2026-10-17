import Init
set_option autoImplicit false
set_option linter.unusedVariables false

def lf_def_9334ffebf4b2148267dba001b091026ed5b6a66f9c40c8b9fd247c7cedd65722 (p0 p1 p2 : Prop) : Prop := p2
theorem lf_lib_9334ffebf4b2148267dba001b091026ed5b6a66f9c40c8b9fd247c7cedd65722 (p0 p1 p2 : Prop)
    (h0 : p0)
    (h1 : (p0 → p1))
    (h2 : (p1 → p2))
    : lf_def_9334ffebf4b2148267dba001b091026ed5b6a66f9c40c8b9fd247c7cedd65722 p0 p1 p2 :=
  (h2 ((h1 (h0))))
#print axioms lf_def_9334ffebf4b2148267dba001b091026ed5b6a66f9c40c8b9fd247c7cedd65722
#print axioms lf_lib_9334ffebf4b2148267dba001b091026ed5b6a66f9c40c8b9fd247c7cedd65722

def lf_def_94501138590923a501a6cc26ee2cb7df72183e51bb2ecf0ab0885f1a268d3fde (p0 p1 : Prop) : Prop := (p0 ∧ p1)
theorem lf_lib_94501138590923a501a6cc26ee2cb7df72183e51bb2ecf0ab0885f1a268d3fde (p0 p1 : Prop)
    (h0 : p0)
    (h1 : p1)
    : lf_def_94501138590923a501a6cc26ee2cb7df72183e51bb2ecf0ab0885f1a268d3fde p0 p1 :=
  (And.intro (h0) (h1))
#print axioms lf_def_94501138590923a501a6cc26ee2cb7df72183e51bb2ecf0ab0885f1a268d3fde
#print axioms lf_lib_94501138590923a501a6cc26ee2cb7df72183e51bb2ecf0ab0885f1a268d3fde

def lf_def_7336460e22551e875fb6202e3cc7ca97c1985659e2896c9b7a30edee14d0e01c (p0 p1 : Prop) : Prop := p0
theorem lf_lib_7336460e22551e875fb6202e3cc7ca97c1985659e2896c9b7a30edee14d0e01c (p0 p1 : Prop)
    (h0 : (p0 ∧ p1))
    : lf_def_7336460e22551e875fb6202e3cc7ca97c1985659e2896c9b7a30edee14d0e01c p0 p1 :=
  (And.left (h0))
#print axioms lf_def_7336460e22551e875fb6202e3cc7ca97c1985659e2896c9b7a30edee14d0e01c
#print axioms lf_lib_7336460e22551e875fb6202e3cc7ca97c1985659e2896c9b7a30edee14d0e01c

def lf_def_e662f8e0d0119aaec891ec51c9fb7f8de632fa6ccbb8b5c918ccd8962d6a881d (p0 p1 p2 p3 : Prop) : Prop := p3
theorem lf_lib_e662f8e0d0119aaec891ec51c9fb7f8de632fa6ccbb8b5c918ccd8962d6a881d (p0 p1 p2 p3 : Prop)
    (h0 : p0)
    (h1 : (p0 → p1))
    (h2 : (p1 → p2))
    (h3 : (p2 → p3))
    : lf_def_e662f8e0d0119aaec891ec51c9fb7f8de632fa6ccbb8b5c918ccd8962d6a881d p0 p1 p2 p3 :=
  (h3 ((h2 ((h1 (h0))))))
#print axioms lf_def_e662f8e0d0119aaec891ec51c9fb7f8de632fa6ccbb8b5c918ccd8962d6a881d
#print axioms lf_lib_e662f8e0d0119aaec891ec51c9fb7f8de632fa6ccbb8b5c918ccd8962d6a881d

def lf_def_0f103a03f9f36ddb4beb8c364afc6c047e2fd66b72938d59bd20008607ed1b89 (p0 p1 p2 : Prop) : Prop := (p0 ∧ (p1 ∧ p2))
theorem lf_lib_0f103a03f9f36ddb4beb8c364afc6c047e2fd66b72938d59bd20008607ed1b89 (p0 p1 p2 : Prop)
    (h0 : p0)
    (h1 : p1)
    (h2 : p2)
    : lf_def_0f103a03f9f36ddb4beb8c364afc6c047e2fd66b72938d59bd20008607ed1b89 p0 p1 p2 :=
  (And.intro (h0) ((And.intro (h1) (h2))))
#print axioms lf_def_0f103a03f9f36ddb4beb8c364afc6c047e2fd66b72938d59bd20008607ed1b89
#print axioms lf_lib_0f103a03f9f36ddb4beb8c364afc6c047e2fd66b72938d59bd20008607ed1b89

def lf_def_ab39828d833e47ad8b7fa39247f7de4d9967c2453e35783a2d035de5efc3018e (p0 p1 p2 : Prop) : Prop := p0
theorem lf_lib_ab39828d833e47ad8b7fa39247f7de4d9967c2453e35783a2d035de5efc3018e (p0 p1 p2 : Prop)
    (h0 : (p0 ∧ (p1 ∧ p2)))
    : lf_def_ab39828d833e47ad8b7fa39247f7de4d9967c2453e35783a2d035de5efc3018e p0 p1 p2 :=
  (And.left (h0))
#print axioms lf_def_ab39828d833e47ad8b7fa39247f7de4d9967c2453e35783a2d035de5efc3018e
#print axioms lf_lib_ab39828d833e47ad8b7fa39247f7de4d9967c2453e35783a2d035de5efc3018e

def lf_def_cf3b892fafedbae9c5b970ed9b2095c4e28ace60983b9633a383612237e75aee (p0 p1 p2 p3 : Prop) : Prop := (p0 ∧ (p1 ∧ (p2 ∧ p3)))
theorem lf_lib_cf3b892fafedbae9c5b970ed9b2095c4e28ace60983b9633a383612237e75aee (p0 p1 p2 p3 : Prop)
    (h0 : p0)
    (h1 : p1)
    (h2 : p2)
    (h3 : p3)
    : lf_def_cf3b892fafedbae9c5b970ed9b2095c4e28ace60983b9633a383612237e75aee p0 p1 p2 p3 :=
  (And.intro (h0) ((lf_lib_0f103a03f9f36ddb4beb8c364afc6c047e2fd66b72938d59bd20008607ed1b89 (p1) (p2) (p3) (h1) (h2) (h3))))
#print axioms lf_def_cf3b892fafedbae9c5b970ed9b2095c4e28ace60983b9633a383612237e75aee
#print axioms lf_lib_cf3b892fafedbae9c5b970ed9b2095c4e28ace60983b9633a383612237e75aee

