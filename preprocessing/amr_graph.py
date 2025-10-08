from __future__ import annotations

import re
import time
from typing import Any

import amrlib

import warnings
from igraph import Graph
import penman
from nltk.corpus import propbank

from preprocessing.utils import query_conceptnet, find_wd_id

stog = amrlib.load_stog_model()


class Entity:
    SUCCESSFUL_LOOKUPS_COUNTER = 0
    LOOKUP_ERROR_COUNTER = 0

    def __init__(self, var: str, entity: str, id: str = None):
        """
        Entity or a Literal in a Triple.
        :param var: variable name in AMR (e.g., b, b1, b2)
        :param entity: entity to be represented
        :param id: Wikidata Q-ID if available (usually added later, if found)
        """
        self.var, self.entity = var, entity
        # "concept" (belongs to a KG), "literal", "negation" ('polarity -'), "unknown" (for questions)
        self.type = self.define_type()
        self.unique = self.entity
        clean_match = re.search(r"([a-zA-Z]+)(?:[-\"]?\d+)?$", self.entity)
        self.clean = clean_match.group(1) if clean_match else self.entity
        self.id = id  # Wikidata ID

        self.prefix = self.define_prefix() # "amr", "pb" (PropBank), "wd" (Wikidata), "cn" (ConceptNet)
        if self.prefix == "pb":
            self.args = self.get_args()
        else:
            self.args = None

    def __repr__(self):
        return (f"Entity(var={self.var}, entity={self.entity}, unique={self.unique}, "
                f"type={self.type}, prefix={self.prefix}, id={self.id})")

    def define_type(self) -> str:
        """
        Define the type of the entity based on its characteristics.
        """
        if self.is_literal():
            self.entity = self.entity.strip('"')
            return "literal"
        elif self.is_negation():
            return "negation"
        else:
            return "concept"

    def is_negation(self) -> bool:
        """
        Check if the entity represents a negation (polarity "-").
        """
        return self.entity == "-"

    def get_resourse(self) -> str:
        """
        Get the full resource identifier for the entity.
        :return: full resource identifier
        """
        if self.id:
            return f"{self.prefix}:{self.id}"
        elif self.type == "negation":
            # always use "amr" prefix for negation
            return f"{self.prefix}:negative"  # for polarity "-"
        elif self.type == "literal":
            return self.unique
        else:
            if self.prefix not in ("pb", "cn", "amr"):
                raise ValueError(f"Unexpected resourse option for entity: {self}")
            return f"{self.prefix}:{self.unique.replace(' ', '_')}"

    def define_prefix(self) -> str:
        """
        Define prefix to the entity based on its type.
        """
        if self.type == "literal":
            return ""
        elif self.id:
            return "wd"
        if self.is_pb_concept():
            return "pb"
        elif self.in_conceptnet() and self.entity != "and":
            return "cn"
        else:
            return "amr"

    def var_is_repeated(self) -> bool:
        """
        Check if word is a repeated variable - contains a disambiguating number.
        """
        return bool(re.search(r"[-a-zA-Z]*[a-zA-Z]\d$", self.var))

    def is_pb_concept(self) -> bool:
        """
        Check if a word is a PropBank concept (i.e., contains a dash and a number at the end).
        """
        return bool(re.search(r"-\d+$", self.entity))

    def is_decimal_fraction(self) -> bool:
        """
        Check if word is a decimal fraction.
        """
        return bool(re.match(r"^\d+[,/.]\d+$", self.entity))

    def is_literal(self) -> bool:
        """
        Check if the entity is a literal.
        """
        quote = '"' in self.entity
        if self.entity.isnumeric() or self.is_decimal_fraction() or quote:
            return True
        return False

    def in_conceptnet(self) -> bool:
        """
        Check if a word is in ConceptNet.
        """
        response = query_conceptnet(self.entity)
        return "edges" in response and len(response["edges"]) > 0

    def get_args(self) -> dict:
        """
        Get the argument roles for a given concept from PropBank.
        # TODO: switch to using a local PropBank dataset, as nltk's version is outdated
        :return: dictionary mapping argument labels to their descriptions
        """
        concept = self.entity.replace("-", ".")
        print("Looking up roles for concept:", concept)
        try:
            rs = propbank.roleset(concept)
            concept_args = {}
            for r in rs.findall("roles/role"):
                # handles multiple roles, e.g. secondary attribute, described-as => secondary_attribute_or_described-as
                descriptor = r.attrib['descr'].replace(' ', '_').replace(",", "_or")
                concept_args[f"ARG{r.attrib['n']}"] = descriptor.lower()
            Entity.SUCCESSFUL_LOOKUPS_COUNTER += 1
            return concept_args
        except ValueError as e:
            Entity.LOOKUP_ERROR_COUNTER += 1
            warnings.warn(f"PropBank lookup failed: {e}")
            return {}


class Relation:
    def __init__(self, text: str, prefix: str = "amr"):
        """
        Relation in a Triple.
        :param text: relation text (e.g., :mod, :ARG0)
        :param prefix: prefix for the relation (default: "amr", changed in the course of processing)
        """
        self.text = text.strip(":")
        if self.is_option():
            self.descriptor = self.text.replace("op", "option")
        else:
            self.descriptor = self.text # in case of mapping to PropBank roles
        self.prefix = prefix  # can be changed to "pb" for PropBank roles

    def __repr__(self):
        return f"Relation(text={self.text}, descriptor={self.descriptor}, prefix={self.prefix})"

    def is_option(self) -> bool:
        """
        Check if the relation is an option (e.g., :op1, :op2).
        Often, it is used to represent parts of a name/title after :name relations.
        """
        return bool(re.match(r"op\d+", self.text))

    def is_name(self) -> bool:
        """
        Check if the relation is a name relation (:name).
        """
        return self.text == "name"

    def get_resource(self) -> str:
        """
        Get the full resource identifier for the relation.
        :return: full resource identifier
        """
        return f"{self.prefix}:{self.descriptor}"


class Triple:
    def __init__(self, head: Entity | str, relation: Relation, tail: Entity | str):
        """
        A triple in the AMR graph.
        :param head: Entity object; can only be a string for wd:Q-ids
        :param relation: Relation object
        :param tail: Entity object; can only be a string for literals or wd:Q-ids
        """
        # if isinstance(head, str) and not head.startswith("wd:"):
        #     raise ValueError(f"Head must be an Entity or a wd-id string, got: {head}")
        self.head = head
        self.relation = relation
        self.tail = tail

        # Original triple before any processing
        self.before = (
            head.unique if isinstance(head, Entity) else head,
            relation.text,
            tail.unique if isinstance(tail, Entity) else tail
        )
        self.after = None

    def __repr__(self):
        return f"Triple(head={self.head}, relation={self.relation}, tail={self.tail})"

    def finalize(self) -> Triple:
        """
        Finalize the triple by setting the 'after' attribute if not already set.
        :return: the Triple object itself
        """
        # There are cases when 'after' is set to empty tuple during preprocessing to mark removed triples
        if self.after is None:
            head = self.head if isinstance(self.head, str) else self.head.get_resourse()
            relation = self.relation if isinstance(self.relation, str) else self.relation.get_resource()
            tail = self.tail if isinstance(self.tail, str) else self.tail.get_resourse()
            self.after = (head, relation, tail)
        return self


class AMRGraph:
    PROCESSING_TIMES = []

    @staticmethod
    def print_ave_time():
        """ Print the average processing time of all AMR graphs created so far. """
        if not AMRGraph.PROCESSING_TIMES:
            print(" No AMR graphs processed yet.")
            return
        average_time = sum(AMRGraph.PROCESSING_TIMES) / len(AMRGraph.PROCESSING_TIMES)
        print(" Average processing time: ", f"{average_time:.2f} seconds.")

    def __init__(self, sentence: str, verbose: bool = False):
        """
        Create an AMR-based graph from a sentence.
        It is instantly post-processed to identify entities and
        map them to Wikidata Q-IDs and represented with igraph.
        :param sentence: input sentence
        :param verbose: whether to print detailed processing information
        """
        start_time = time.time()
        self.verbose = verbose

        self.sentence = sentence.strip()
        _, self.amr_graph = stog.parse_sents([self.sentence])[0].split("\n", 1)
        self.amr_triples = penman.decode(self.amr_graph).triples

        if self.verbose:
            self._print(self.amr_triples, f"AMR Triples ({len(self.amr_triples)})")

        self.entities = self.disambiguate_vars(self.extract_entities())

        if self.verbose:
            self._print(self.entities, f"Extracted entities ({len(self.entities)})")

        self.raw_triples = []
        self.term_map = None
        self.triples = self.make_triples()

        if self.verbose:
            self._print(self.triples, f"Triples ({len(self.triples)})", before_after=True)

        self.graph: Graph = self.create_graph()
        if self.verbose:
            print("Graph summary:")
            print(" ", self.graph.summary(1))

        end_time = time.time()
        processing_time = end_time - start_time
        AMRGraph.PROCESSING_TIMES.append(processing_time)
        if self.verbose:
            print("-"*30, f"Processed AMR graph in {processing_time:.2f} seconds.", sep="\n")

    def _print(self, data: list[Any], title: str, before_after: bool = False) -> None:
        print(f"{title}:")
        for point in data:
            if before_after and hasattr(point, "before") and hasattr(point, "after"):
                print(" ", point.before, "=>", point.after)
            else:
                print(" ", point)

    def apply_wd_terms(self):
        """
        Replace entities in triples with their identified terms from Wikidata.
        :return: None
        """
        for triple in self.raw_triples:
            head_in_term = triple.head in self.term_map
            tail_in_term = triple.tail in self.term_map
            if head_in_term and tail_in_term:
                triple.after = ()
            elif head_in_term:
                triple.head = self.term_map[triple.head]
            elif tail_in_term:
                triple.tail = self.term_map[triple.tail]

        for ent in self.entities:
            if not ent.id:
                continue
            entity = ent.entity.replace(' ', '_')
            prefix = "cn" if ent.in_conceptnet() else "amr"
            t1 = Triple(f"{prefix}:{entity}", Relation("sameAs", "owl"), f"wd:{ent.id}")
            t2 = Triple(f"wd:{ent.id}", Relation("label", "rdfs"), ent)
            for t in [t1, t2]:
                if self.verbose:
                    print(f"Adding extra triple: {t}")
                # Make sure these are marked as added triples
                t.finalize()
                t.after = t.before
                t.before = ()
                self.raw_triples.append(t)

    def create_term_map(self) -> dict[Entity, Entity]:
        """
        Create a mapping of entities to their identified terms with Wikidata Q-IDs.
        :return: dictionary mapping Entity objects to their identified terms
        """
        self.term_map = {}
        for i, triple in enumerate(self.raw_triples, 0):
            head, tail = triple.head, triple.tail
            if type(tail) is str:
                print(f"Triple with string tail: {triple}")
            if tail.type == "concept" and "pb" not in (head.prefix, tail.prefix):
                term = self.identify_term(triple)
                if not term:
                    continue
                self.term_map[triple.head] = term
                self.term_map[triple.tail] = term
        return self.term_map

    def identify_term(self, triple: Triple) -> None | Entity:
        """
        Identify compound nouns / terms in the AMR triples and find their Wikidata Q-ID.
        :param triple: Triple object
        :return: Entity object with Q-ID or None if not found
        """
        head, rel, tail = triple.head, triple.relation, triple.tail
        possible_term = ""
        if rel.text in ["mod", "source"]:
            possible_term = f"{tail.clean} {head.clean}"
        if rel.text == "poss":
            possible_term = f"{head.clean} of {tail.clean}"
        if rel.text.endswith("ARG1"):
            possible_term = f"{head.clean} {tail.clean}"
            warnings.warn(f"ARG1 text, possible term: {possible_term} (check the necessity)")
        if not possible_term:
            return None
        wd_id = find_wd_id(possible_term)
        if not wd_id:
            return None
        if self.verbose:
            print(f"Found wd-id for term '{possible_term}': {wd_id}")
        return self.create_entity(possible_term, wd_id)

    def make_triples(self) -> list[Triple]:
        """
        Create Triple objects from AMR triples.
        :return: list of Triple objects
        """
        var_to_entity = {ent.var: ent for ent in self.entities}
        # amr triples now have variable definitions removed
        for head, rel, tail in self.amr_triples:
            if rel == ":instance":
                continue
            head = var_to_entity[head]
            rel = Relation(rel)
            if head.prefix == "pb":
                rel.prefix = "pb"
                # some relations might stay as ARGn even if the concept was retrieved from PropBank,
                # if the source list of args is not exhaustive
                rel.descriptor = head.args.get(rel.text, rel.text)
            # create a literal entity if not found; if the function is in 'get' adds one regardless
            tail = var_to_entity.get(tail) or self.create_entity(tail)
            triple = Triple(head, rel, tail)
            self.raw_triples.append(triple)

        # Collect names into single literals and adjust triples
        self.aggregate_names()
        # Create a term map for entities with Q-IDs to replace them in triples
        self.create_term_map()
        # Replace entities in triples with their identified terms from Wikidata
        self.apply_wd_terms()
        return [t.finalize() for t in self.raw_triples]

    def create_entity(self, entity_parts: list[str | Entity] | str, wd_id: str = None) -> Entity:
        """
        Create a name or term entity from parts of a name or single string.
        :param entity_parts: list of parts of name or single string
        :param wd_id: optional Wikidata Q-ID
        :return: Entity object
        """
        if isinstance(entity_parts, list):
            # If the entity is already created and exists in the list, return it
            if len(entity_parts) == 1 and isinstance(entity_parts[0], Entity):
                if self.verbose:
                    print(f"Entity already exists: {entity_parts[0]}")
                return entity_parts[0]
            parts = [p if isinstance(p, str) else p.entity for p in entity_parts]
            full_string = " ".join(parts)
        else:
            full_string = entity_parts
        entity = Entity("", full_string, id=wd_id)
        if isinstance(entity_parts[0], Entity) and not wd_id:
            entity.type = entity_parts[0].type
        if self.verbose:
            print(f"Adding extra entity: {entity}")
        self.entities.append(entity)
        return entity

    def aggregate_names(self):
        """
        Aggregate fragmented names from triples based on :name and :op relations into literals.
        :return: list of names
        """
        name_triple = None
        name = []
        for t in self.raw_triples:
            head, rel, tail = t.head, t.relation, t.tail
            if rel.is_name():
                if name and name_triple:
                    name_triple.tail = self.create_entity(name)
                    name = []
                name_triple = t # the full name will be assigned to the tail
                continue
            if rel.is_option() and name_triple:
                name.append(tail)
                t.after = () # remove the :op triples from the final list
            elif name and name_triple:
                name_triple.tail = self.create_entity(name)
                name, name_triple = [], None
        if name and name_triple:
            name_triple.tail = self.create_entity(name)

    def extract_entities(self) -> list[Entity]:
        """
        Extract entities from AMR triples based on :instance relations.
        The variable definitions are removed from the original triples.
        :return: dictionary mapping variable names to Entity objects
        """
        entities = []
        for i, triple in enumerate(self.amr_triples):
            subj, rel, obj = triple
            if rel == ":instance":
                entities.append(Entity(var=subj, entity=obj))
        return entities

    @staticmethod
    def disambiguate_vars(entities: list[Entity]) -> list[Entity]:
        """
        Disambiguate repeated variables if they are not concepts and appear multiple times.
        :param variable_map: mapping of variables to concepts
        :return: updated variable_map with disambiguated variables
        """
        for i, ent in enumerate(entities):
            ent_count = sum(1 for e in entities if e.entity == ent.entity)
            if ent.var_is_repeated() and ent_count > 1 and ent.prefix != "pb":
                number = ent.var[-1]
                ent.unique = ent.entity + number
        return entities

    def create_graph(self):
        """
        Creates a graph from given triples
        :param triples: preprocessed triples
        :return: an igraph instance
        """
        g = Graph(directed=True)
        triples = [t.after for t in self.triples if t.after]  # Remove empty tuples
        subj_nodes = [head for head, _, _ in triples]
        obj_nodes = [tail for _, _, tail in triples]
        # disregard the warning about a graph missing
        g.add_vertices(list(set(subj_nodes + obj_nodes)))
        for head, rel, tail in triples:
            g.add_edge(head, tail, label=rel)
        return g


if __name__ == "__main__":
    # from amr_examples import AMR_EXAMPLES
    #
    # # TODO: change index to test different examples
    # i = 0
    # if i >= len(AMR_EXAMPLES):
    #     raise ValueError(f"Index {i} out of range for AMR_EXAMPLES with length {len(AMR_EXAMPLES)}")
    # for j in range(i, len(AMR_EXAMPLES)):
    #     print(f"\nExample {j+1}:")
    #     sentence = AMR_EXAMPLES[j]
    #     print(sentence, "-"*30, sep="\n")
    #     graph = AMRGraph(sentence, verbose=True)
    # AMRGraph.print_ave_time() # for test sentences: 6.52 seconds
    # # The number of unresolved PropBank ARGs is a bit higher than the lookup error count,
    # # as some retrieved concepts do not have ARGs produced by the model (probably due to nltk being outdated)
    # print(f"PropBank lookup errors: "
    #       f"{Entity.LOOKUP_ERROR_COUNTER} out of {Entity.SUCCESSFUL_LOOKUPS_COUNTER} concepts")

    sentences = [
        "Coburg Peak is the rocky peak rising to 783 m in Erul Heights on Trinity Peninsula in Graham Land, Antarctica.",
        "It is surmounting Cugnot Ice Piedmont to the northeast.",
        "The peak is named after the Bulgarian royal house of Coburg (Saxe-Coburg-Gotha), 1887–1946."
    ]
    graph = AMRGraph(" ".join(sentences), verbose=True)



