import crypten
import crypten.communicator as comm

def encrypt_tensor(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()
    assert world_size  == 2
    
    # assumes party 1 is the actual data provider
    src_id = 1

    if rank == src_id:
        input_upd = input.cuda()
    else:
        input_upd = torch.empty(input.size()).cuda()
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input

def encrypt_model(model, modelFunc, config):
    rank = comm.get().get_rank()
    
    # assumes party 0 is the actual model provider
    if rank == 0:
        model_upd = model
    else:
        if isinstance(config, tuple):
            model_upd = modelFunc(*config).cuda()
        else:
            model_upd = modelFunc(config).cuda()

    private_model = model_upd.encrypt(src=0)
    return private_model
