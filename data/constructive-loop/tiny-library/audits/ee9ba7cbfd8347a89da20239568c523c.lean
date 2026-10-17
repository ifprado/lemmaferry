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

