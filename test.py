from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer


def run_test(epoch=-1, name="", is_train=True):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name
    dataset = DataLoader(opt, is_train)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        if opt.use_block:
            ncorrect_tmp = 0
            for j in range(len(ncorrect)):
                ncorrect_tmp += ncorrect[j]
            ncorrect_tmp /= len(ncorrect)
            ncorrect = ncorrect_tmp
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
