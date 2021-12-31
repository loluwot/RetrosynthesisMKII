from ord_schema.proto import dataset_pb2
from ord_schema.proto import reaction_pb2 
import gzip
TRAINING_PATH = './TRAINING_DATA/'

ROLE_TYPES = reaction_pb2.ReactionRole.ReactionRoleType
IDENTIFIER_TYPES = reaction_pb2.CompoundIdentifier.IdentifierType

def get_smiles(compound):
    for identifier in compound.identifiers:
        if identifier.type == IDENTIFIER_TYPES.SMILES:
            return identifier.value
    return -1

def get_reaction_smiles(reaction):
    for identifier in reaction.identifiers:
        if identifier.type == reaction_pb2.ReactionIdentifier.IdentifierType.REACTION_CXSMILES:
            return identifier.value
    return -1
    
def file_to_rxns(file, additional_info=False):
    ds = dataset_pb2.Dataset()
    ds.ParseFromString(gzip.open(file, 'rb').read())
    if additional_info:
        return list(zip(ds.reactions, [file]*len(ds.reactions)), range(len(ds.reactions)))
    else:
        return ds.reactions
