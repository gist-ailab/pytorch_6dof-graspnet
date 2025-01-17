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
            # print('i>>>>>>>', i)
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                if not opt.cross_condition:
                    loss_types = []
                    if opt.arch == "vae":
                        loss = [
                            model.loss, model.kl_loss, model.reconstruction_loss,
                            model.confidence_loss, model.angle_loss
                        ]
                        loss_types = [
                            "total_loss", "kl_loss", "reconstruction_loss",
                            "confidence loss", "angle_loss"
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
                else:
                    loss1 = [model.loss[0], model.kl_loss[0], model.reconstruction_loss[0], model.confidence_loss[0]]
                    loss21 = [model.loss[1], model.kl_loss[1], model.reconstruction_loss[1], model.confidence_loss[1]]
                    loss22 = [model.loss[2], model.kl_loss[1], model.reconstruction_loss[2], model.confidence_loss[2]]
                    loss1_types = [
                            "total_loss1", "kl_loss1", "reconstruction_loss1",
                            "confidence loss1"]
                    loss21_types = [
                            "total_loss21", "kl_loss21", "reconstruction_loss21",
                            "confidence loss21"]
                    loss22_types = [
                            "total_loss22", "kl_loss22", "reconstruction_loss22",
                            "confidence loss22"]
                    
                    t = (time.time() - iter_start_time) / opt.batch_size
                    writer.print_current_losses(epoch, epoch_iter, loss1, t, t_data, loss1_types)
                    writer.print_current_losses(epoch, epoch_iter, loss21, t, t_data, loss21_types)
                    writer.print_current_losses(epoch, epoch_iter, loss22, t, t_data, loss22_types)
                    writer.plot_loss(loss1, epoch, epoch_iter, dataset_size, loss1_types)
                    writer.plot_loss(loss21, epoch, epoch_iter, dataset_size, loss21_types)
                    writer.plot_loss(loss22, epoch, epoch_iter, dataset_size, loss22_types)
                    
                
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
