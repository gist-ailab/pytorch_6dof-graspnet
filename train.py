import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer
from test import run_test
import threading
import numpy as np
import open3d as o3d

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,4,5'

def main():
    opt = TrainOptions().parse()

    ##* for debug
    
    
    if opt == None:
        return

    dataset = DataLoader(opt)
    if opt.is_bimanual:
        dataset_size = len(dataset) * opt.num_grasps_per_object
    else:    
        dataset_size = len(dataset) * opt.num_grasps_per_object

    
    model = create_model(opt)

    writer = Writer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            
            ##############*data visualization*################
            # print(data['pc'].shape)
            # print(data['cad_path'][0])
            # if i == 0:
            #     continue
            # pcd_object = o3d.geometry.PointCloud()
            # pcd_object.points = o3d.utility.Vector3dVector(data['pc'][0])
            
            # target_cps = data['target_cps']
            # target_cps_list = []
            
            # for i in range(len(target_cps)):
            #     cps = target_cps[i]
            #     pcd_target_cps = o3d.geometry.PointCloud()
            #     pcd_target_cps.points = o3d.utility.Vector3dVector(cps)
            #     pcd_target_cps.paint_uniform_color([0, 0, 0])
            #     target_cps_list.append(pcd_target_cps)
            # # for i in range(len(target_cps)):
            # #     o3d.visualization.draw_geometries([pcd_object]+[target_cps_list[i]])
            # o3d.visualization.draw_geometries([pcd_object]+target_cps_list)
            
            # exit()
            
            ################*
            
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                loss_types = []
                if opt.arch == "vae":
                    loss = [
                        model.loss, model.kl_loss, model.reconstruction_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "kl_loss", "reconstruction_loss",
                        "confidence loss"
                    ]
                elif opt.arch == "gan":
                    loss = [
                        model.loss, model.reconstruction_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "reconstruction_loss", "confidence_loss"
                    ]
                else:
                    loss = [
                        model.loss, model.classification_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "classification_loss", "confidence_loss"
                    ]
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data,
                                            loss_types)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size,
                                 loss_types)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest', epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest', epoch)
            model.save_network(str(epoch), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,
               time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch, name=opt.name, is_train=False)
            writer.plot_acc(acc, epoch)

    writer.close()


if __name__ == '__main__':
    main()
